# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for CoState: UnifiedPrefixDirectory, HostHSCache, OnlineCostGate, and connector integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeLookupResult,
    TailKeyBoundary,
)

from vllm_omni.distributed.kv_transfer.mooncake.config import MooncakeConfig
from vllm_omni.distributed.kv_transfer.mooncake.costate import (
    COSTATE_HS_SENTINEL_GROUP_ID,
    CoStateWorkerMetadata,
    HostHSCache,
    OnlineCostGate,
    UnifiedPrefixDirectory,
    is_text_only_request,
    split_costate_lookup_result,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_cfg(**overrides) -> MooncakeConfig:
    base = {
        "config_file": "/tmp/mc.json",
        "namespace": "test-ns",
        "model_revision": "rev1",
        "max_hs_bytes": 64 * 1024 * 1024,
        "max_pending_transfers": 8,
        "enable_costate": True,
        "enable_local_tier": True,
        "enable_modality_decouple": True,
        "enable_cost_gate": True,
        "min_write_hits": 2,
        "eviction_policy": "costate_utility",
        "sync_write_truncation": True,
        "max_store_bytes": 0,
        "host_hs_cache_bytes": 8 * 1024 * 1024,
    }
    base.update(overrides)
    return MooncakeConfig.from_dict(base)


def _hashes(prefix: str, n: int) -> list[bytes]:
    return [f"{prefix}-blk-{i:04d}".encode().ljust(32, b"0") for i in range(n)]


def test_is_text_only_request_detection() -> None:
    req_audio = SimpleNamespace(additional_information={"omni_final_stage_id": 2})
    req_text = SimpleNamespace(additional_information={"omni_final_stage_id": 0})
    req_forced = SimpleNamespace(additional_information={"omni_final_stage_id": 0, "omni_force_kv_transfer": True})
    req_none = SimpleNamespace(additional_information=None)

    assert not is_text_only_request(req_audio)
    assert is_text_only_request(req_text)
    assert not is_text_only_request(req_forced)
    assert not is_text_only_request(req_none)


def test_split_costate_lookup_result_strips_sentinel() -> None:
    raw = MooncakeLookupResult(
        2048,
        (
            TailKeyBoundary(group_id=0, num_tokens=2048),
            TailKeyBoundary(group_id=COSTATE_HS_SENTINEL_GROUP_ID, num_tokens=512),
        ),
    )
    kv_hit, joint_hit, clean = split_costate_lookup_result(raw)
    assert kv_hit == 2048
    assert joint_hit == 512
    assert clean == (TailKeyBoundary(group_id=0, num_tokens=2048),)


def test_host_hs_cache_lru_and_prefix_assembly() -> None:
    # Each block: 2 * 16 * 64 * 2 bytes = 4096 bytes. Cap at 3 blocks (12288 bytes).
    cache = HostHSCache(max_bytes=12288)
    keys = _hashes("doc", 4)
    blocks = [
        {
            "embedding": torch.full((16, 64), float(i), dtype=torch.bfloat16),
            "hidden": torch.full((16, 64), float(i + 10), dtype=torch.bfloat16),
        }
        for i in range(4)
    ]

    for i in range(3):
        ok, ev = cache.put_block(keys[i], blocks[i])
        assert ok and ev == []

    # Touch block 0 so block 1 becomes LRU victim
    assert cache.get_block(keys[0]) is not None
    ok, ev = cache.put_block(keys[3], blocks[3])
    assert ok and ev == [keys[1]]
    assert not cache.contains(keys[1])

    # Assembling [0, 2, 3] succeeds with exact 48 rows
    merged = cache.assemble_prefix([keys[0], keys[2], keys[3]])
    assert merged is not None
    assert merged["embedding"].shape == (48, 64)
    assert float(merged["embedding"][0, 0]) == 0.0
    assert float(merged["embedding"][16, 0]) == 2.0
    assert float(merged["embedding"][32, 0]) == 3.0


def test_directory_two_hit_admission_and_idempotent_visit() -> None:
    cfg = _make_cfg(min_write_hits=2)
    directory = UnifiedPrefixDirectory(
        block_size=16,
        kv_block_bytes=1_572_864,
        hs_block_bytes=131_520,
        settings=cfg,
    )
    chain = _hashes("cold", 8)

    # Async lookup retry on the SAME req_id must not double-count as 2 hits!
    counts1 = directory.record_request_visit("req-1", chain, is_text_only=False)
    counts1_retry = directory.record_request_visit("req-1", chain, is_text_only=False)
    assert counts1 == [1] * 8
    assert counts1_retry == [1] * 8

    save_kv, save_hs, evicts = directory.plan_store_write_and_evictions(
        block_hashes=chain, target_tokens=128, is_text_only=False, avail_hs_blocks=100
    )
    assert (save_kv, save_hs, evicts) == (0, 0, [])
    assert directory.cold_filtered_requests == 1

    # Second distinct request sharing first 5 blocks materializes those 5 blocks
    chain2 = chain[:5] + _hashes("tail2", 3)
    directory.record_request_visit("req-2", chain2, is_text_only=False)
    save_kv2, save_hs2, _ = directory.plan_store_write_and_evictions(
        block_hashes=chain2, target_tokens=128, is_text_only=False, avail_hs_blocks=100
    )
    assert save_kv2 == 5 * 16
    assert save_hs2 == 5 * 16

    # Text-only request with modality decoupling writes KV only (save_hs == 0)
    text_chain = _hashes("txt", 4)
    directory.record_request_visit("t-1", text_chain, is_text_only=True)
    directory.record_request_visit("t-2", text_chain, is_text_only=True)
    tkv, ths, _ = directory.plan_store_write_and_evictions(
        block_hashes=text_chain, target_tokens=64, is_text_only=True, avail_hs_blocks=100
    )
    assert tkv == 64
    assert ths == 0


def test_directory_bottom_up_utility_eviction_preserves_shared_roots() -> None:
    # 1 block (KV+HS) = 1000 + 100 = 1100 bytes; allow max 6 blocks = 6600 bytes.
    cfg = _make_cfg(min_write_hits=1, max_store_bytes=6600, eviction_policy="costate_utility")
    directory = UnifiedPrefixDirectory(
        block_size=16,
        kv_block_bytes=1000,
        hs_block_bytes=100,
        settings=cfg,
    )
    shared_root = _hashes("shared-root", 2)
    hot_branch = shared_root + _hashes("hot-leaf", 2)
    cold_branch = shared_root + _hashes("cold-leaf", 2)

    # Cold branch visited once
    directory.record_request_visit("r-cold", cold_branch, is_text_only=False)
    directory.plan_store_write_and_evictions(
        block_hashes=cold_branch, target_tokens=64, is_text_only=False, avail_hs_blocks=50
    )

    # Hot branch visited 4 times (fills store to 2 shared + 2 cold + 2 hot = 6 blocks = 6600B)
    for idx in range(4):
        directory.record_request_visit(f"r-hot-{idx}", hot_branch, is_text_only=False)
    directory.plan_store_write_and_evictions(
        block_hashes=hot_branch, target_tokens=64, is_text_only=False, avail_hs_blocks=50
    )
    assert directory.store_bytes_used == 6600

    # New high-frequency family arrives needing 2 blocks -> must evict cold_branch leaves tail-first,
    # keeping shared_root and hot_branch 100% contiguous and valid!
    new_hot = _hashes("new-hot", 2)
    for idx in range(5):
        directory.record_request_visit(f"r-new-{idx}", new_hot, is_text_only=False)
    skv, shs, evicted = directory.plan_store_write_and_evictions(
        block_hashes=new_hot, target_tokens=32, is_text_only=False, avail_hs_blocks=50
    )
    assert skv == 32 and shs == 32
    assert set(evicted) == set(cold_branch[2:])
    # Hot branch remains 100% contiguous from block 0 to block 3!
    assert directory.contiguous_hs_tokens(hot_branch) == 64


def test_online_cost_gate_modality_and_congestion_decisions() -> None:
    gate = OnlineCostGate(
        block_size=16,
        row_bytes=8192,
        max_hs_bytes=256 * 1024 * 1024,
        max_pending_transfers=8,
    )

    # 1. At 512 tokens (32 blocks), fixed prefill launch amortization beats tiny L2 read -> restore!
    chosen_audio_512 = gate.select_external_tokens(
        local_tokens=0,
        candidate_total_tokens=512,
        is_text_only=False,
        host_hs_covers_local=False,
        enable_cost_gate=True,
    )
    assert chosen_audio_512 == 512

    # 2. At 4096 tokens (256 blocks), 3B-active MoE GPU prefill beats TCP L2 restore -> bypass (zero 4K inversion)!
    chosen_audio_4k = gate.select_external_tokens(
        local_tokens=0,
        candidate_total_tokens=4096,
        is_text_only=False,
        host_hs_covers_local=False,
        enable_cost_gate=True,
    )
    assert chosen_audio_4k == 0

    # 3. At 8192 tokens (512 blocks), quadratic Attention makes idle L2 (c=1) beat cold prefill -> restore;
    #    second concurrent reader sees active_read_count == 1 -> bypasses to prevent TCP contention inversion!
    chosen_8k_first = gate.select_external_tokens(
        local_tokens=0,
        candidate_total_tokens=8192,
        is_text_only=False,
        host_hs_covers_local=False,
        enable_cost_gate=True,
    )
    assert chosen_8k_first == 8192
    gate.reserve_read("req-first", 8192)

    chosen_8k_second = gate.select_external_tokens(
        local_tokens=0,
        candidate_total_tokens=8192,
        is_text_only=False,
        host_hs_covers_local=False,
        enable_cost_gate=True,
    )
    assert chosen_8k_second == 0

    # 4. Single-flight non-interfering L2 write admission
    assert gate.can_admit_store_write(active_gpu_prefills=1)
    assert not gate.can_admit_store_write(active_gpu_prefills=3)
    gate.reserve_save("req-save", hs_tokens=0, kv_tokens=1024)
    assert not gate.can_admit_store_write(active_gpu_prefills=1)
    gate.release_save("req-save")
    assert gate.can_admit_store_write(active_gpu_prefills=1)


def test_host_hs_cache_contiguous_slabs_and_single_pass_suffix() -> None:
    cache = HostHSCache(max_bytes=16 * 1024 * 1024)
    keys = _hashes("slab", 10)
    slab = {
        "embedding": torch.arange(160 * 32, dtype=torch.bfloat16).reshape(160, 32),
        "hidden": torch.arange(160 * 32, dtype=torch.bfloat16).reshape(160, 32) + 1.0,
    }
    added, evicted = cache.put_contiguous_blocks(keys, slab, block_size=16, max_slab_blocks=4)
    assert len(added) == 10 and evicted == []
    suffix = {
        "embedding": torch.full((5, 32), 77.0, dtype=torch.bfloat16),
        "hidden": torch.full((5, 32), 88.0, dtype=torch.bfloat16),
    }
    merged = cache.assemble_with_suffix(keys, prefix_rows=155, suffix=suffix)
    assert merged is not None
    assert merged["embedding"].shape == (160, 32)
    assert torch.equal(merged["embedding"][:155], slab["embedding"][:155])
    assert torch.equal(merged["embedding"][155:], suffix["embedding"])
    assert torch.equal(merged["hidden"][:155], slab["hidden"][:155])
    assert torch.equal(merged["hidden"][155:], suffix["hidden"])


def test_directory_store_tokens_and_l1_skip_guard() -> None:
    cfg = _make_cfg(min_write_hits=2)
    directory = UnifiedPrefixDirectory(
        block_size=16,
        kv_block_bytes=1000,
        hs_block_bytes=100,
        settings=cfg,
    )
    chain = _hashes("l1skip", 4)
    directory.record_request_visit("r1", chain, is_text_only=False)
    directory.record_request_visit("r2", chain, is_text_only=False)
    # If L1 hit skipped computing [0, 64), directory must NOT falsely mark them Store-resident
    skv, shs, _ = directory.plan_store_write_and_evictions(
        block_hashes=chain, target_tokens=64, is_text_only=False, avail_hs_blocks=100, computed_start_tokens=64
    )
    assert (skv, shs) == (0, 0)
    assert directory.contiguous_store_tokens(chain, require_hs=True) == 0
    # Once evicted from L1 and recomputed from token 0, Store write succeeds and directory records it
    skv2, shs2, _ = directory.plan_store_write_and_evictions(
        block_hashes=chain, target_tokens=64, is_text_only=False, avail_hs_blocks=100, computed_start_tokens=0
    )
    assert (skv2, shs2) == (64, 64)
    assert directory.contiguous_store_tokens(chain, require_hs=True) == 64


def test_worker_metadata_aggregation() -> None:
    m1 = CoStateWorkerMetadata(completed_saves={1: 1}, host_hs_added=[b"a"], obs_ext_hs_ms_per_block=[0.5])
    m2 = CoStateWorkerMetadata(completed_saves={1: 1, 2: 1}, host_hs_evicted=[b"b"], store_kv_saved=[b"c"])
    m1.aggregate(m2)
    assert m1.completed_saves == {1: 2, 2: 1}
    assert m1.host_hs_added == [b"a"]
    assert m1.host_hs_evicted == [b"b"]
    assert m1.store_kv_saved == [b"c"]
    assert m1.obs_ext_hs_ms_per_block == [0.5]
