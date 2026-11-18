# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CoState: Modality-, tier-, and cost-aware joint state cache management.

Implements the four components of CoState:
1. ``UnifiedPrefixDirectory``: Multi-tier prefix-tree ledger tracking GPU/Store
   ``KV`` and Host/Store ``HS`` residency, request-idempotent visit sketches,
   and bottom-up (leaf-to-root) utility-driven eviction.
2. ``HostHSCache``: Bounded worker-side CPU tensor LRU cache supplying the 8%
   handoff ``HS`` rows at memory speed when 92% ``KV`` hits in local GPU HBM.
3. ``OnlineCostGate``: Shadow memory ledger and online EMA-calibrated cost
   comparator gating external restores against MoE GPU prefill recomputation.
4. ``CoStateReqPlan`` / ``CoStateWorkerMetadata``: Zero-copy step metadata
   synchronizing Scheduler and Worker across process boundaries.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeLookupResult,
    MooncakeStoreWorkerMetadata,
    TailKeyBoundary,
)

from vllm_omni.engine.serialization import deserialize_additional_information

from .config import MooncakeConfig
from .thinker_captures import CaptureTensors

# Sentinel group ID piggybacked onto LookupKey ZMQ responses so Worker can
# report both KV-only hit length and joint (KV+HS) hit length in one round trip.
COSTATE_HS_SENTINEL_GROUP_ID = 0xC057A7E0


def split_costate_lookup_result(
    result: MooncakeLookupResult,
) -> tuple[int, int, tuple[TailKeyBoundary, ...]]:
    """Return ``(kv_hit_tokens, joint_hit_tokens, clean_boundaries)``."""
    kv_hit = result.hit_length
    joint_hit = kv_hit
    clean: list[TailKeyBoundary] = []
    for boundary in result.tail_key_boundaries:
        if boundary.group_id == COSTATE_HS_SENTINEL_GROUP_ID:
            joint_hit = min(kv_hit, boundary.num_tokens)
        else:
            clean.append(boundary)
    return kv_hit, joint_hit, tuple(clean)


def is_text_only_request(request: Any) -> bool:
    """True when the request terminates at Stage 0 (Thinker) without Talker."""
    payload = getattr(request, "additional_information", None)
    if payload is None:
        return False
    info = deserialize_additional_information(payload) if not isinstance(payload, dict) else payload
    if not isinstance(info, dict):
        return False
    return bool(info.get("omni_final_stage_id") == 0 and not info.get("omni_force_kv_transfer", False))


@dataclass
class CoStateReqPlan:
    """Per-request execution, recovery, and write plan decided by Scheduler."""

    req_id: str
    prompt_len: int
    block_hashes: list[bytes]
    is_text_only: bool = False
    local_kv_tokens: int = 0
    ext_load_tokens: int = 0
    ext_load_hs: bool = True
    save_store_kv_tokens: int = 0
    save_store_hs_tokens: int = 0
    save_host_hs: bool = False
    promote_from_host_hs_tokens: int = 0


@dataclass
class CoStateWorkerMetadata(MooncakeStoreWorkerMetadata):
    """Worker-to-Scheduler step telemetry for CoState residency & EMA rates."""

    host_hs_added: list[bytes] = field(default_factory=list)
    host_hs_evicted: list[bytes] = field(default_factory=list)
    store_kv_saved: list[bytes] = field(default_factory=list)
    store_hs_saved: list[bytes] = field(default_factory=list)
    store_evicted: list[bytes] = field(default_factory=list)
    obs_ext_kv_ms_per_block: list[float] = field(default_factory=list)
    obs_ext_hs_ms_per_block: list[float] = field(default_factory=list)

    def has_payload(self) -> bool:
        return bool(
            self.completed_saves
            or self.host_hs_added
            or self.host_hs_evicted
            or self.store_kv_saved
            or self.store_hs_saved
            or self.store_evicted
            or self.obs_ext_kv_ms_per_block
            or self.obs_ext_hs_ms_per_block
        )

    def aggregate(self, other: MooncakeStoreWorkerMetadata) -> CoStateWorkerMetadata:
        super().aggregate(other)
        if isinstance(other, CoStateWorkerMetadata):
            self.host_hs_added.extend(other.host_hs_added)
            self.host_hs_evicted.extend(other.host_hs_evicted)
            self.store_kv_saved.extend(other.store_kv_saved)
            self.store_hs_saved.extend(other.store_hs_saved)
            self.store_evicted.extend(other.store_evicted)
            self.obs_ext_kv_ms_per_block.extend(other.obs_ext_kv_ms_per_block)
            self.obs_ext_hs_ms_per_block.extend(other.obs_ext_hs_ms_per_block)
        return self


class HostHSCache:
    """Thread-safe bounded CPU tensor LRU cache for 16-token HS blocks."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._lock = threading.RLock()
        self._entries: OrderedDict[bytes, tuple[CaptureTensors, int]] = OrderedDict()
        self._used_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def contains(self, block_hash: bytes) -> bool:
        with self._lock:
            return block_hash in self._entries

    def put_block(self, block_hash: bytes, tensors: CaptureTensors) -> tuple[bool, list[bytes]]:
        """Insert or refresh one block; return ``(stored, evicted_hashes)``."""
        key = bytes(block_hash)
        cloned: CaptureTensors = {
            name: tensor.detach().cpu().contiguous().clone() for name, tensor in tensors.items()
        }
        nbytes = sum(t.numel() * t.element_size() for t in cloned.values())
        if nbytes <= 0 or nbytes > self.max_bytes:
            return False, []
        evicted: list[bytes] = []
        with self._lock:
            if key in self._entries:
                _, old_bytes = self._entries.pop(key)
                self._used_bytes -= old_bytes
            while self._used_bytes + nbytes > self.max_bytes and self._entries:
                ev_key, (_, ev_bytes) = self._entries.popitem(last=False)
                self._used_bytes -= ev_bytes
                self.evictions += 1
                evicted.append(ev_key)
            self._entries[key] = (cloned, nbytes)
            self._used_bytes += nbytes
        return True, evicted

    def get_block(self, block_hash: bytes) -> CaptureTensors | None:
        key = bytes(block_hash)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[0]

    def get_blocks(self, block_hashes: Sequence[bytes]) -> list[CaptureTensors] | None:
        out: list[CaptureTensors] = []
        with self._lock:
            for raw_hash in block_hashes:
                key = bytes(raw_hash)
                entry = self._entries.get(key)
                if entry is None:
                    self.misses += 1
                    return None
                self._entries.move_to_end(key)
                out.append(entry[0])
            self.hits += len(out)
        return out

    def assemble_prefix(self, block_hashes: Sequence[bytes]) -> CaptureTensors | None:
        blocks = self.get_blocks(block_hashes)
        if not blocks:
            return None
        return {name: torch.cat([b[name] for b in blocks], dim=0) for name in ("embedding", "hidden")}


@dataclass
class PrefixNode:
    """One 16-token block node in CoState's Unified Prefix Directory."""

    block_hash: bytes
    depth: int
    parent: PrefixNode | None
    children: dict[bytes, PrefixNode] = field(default_factory=dict)
    host_hs: bool = False
    store_kv: bool = False
    store_hs: bool = False
    access_count: int = 0
    audio_access_count: int = 0
    last_step: int = 0
    active_refs: int = 0

    @property
    def in_store(self) -> bool:
        return self.store_kv or self.store_hs


def marginal_prefill_ms_per_block(depth: int) -> float:
    """Calibrated per-16-token-block prefill cost on Qwen3-Omni-30B-A3B."""
    if depth < 128:
        return 0.38
    if depth < 256:
        return 0.83
    return 1.15


class OnlineCostGate:
    """Mechanism 2: Shadow HS budget ledger + online EMA Load-vs-Recompute gate."""

    def __init__(
        self,
        *,
        block_size: int,
        row_bytes: int,
        max_hs_bytes: int,
        max_pending_transfers: int,
        header_allowance: int = 16392,
    ) -> None:
        self.block_size = block_size
        self.row_bytes = row_bytes
        self.max_hs_bytes = max_hs_bytes
        self.max_pending_transfers = max_pending_transfers
        self.header_allowance = header_allowance
        # Initialized from Wave 1A calibration; adapts via online EMA.
        self.ema_kv_ms_per_block = 0.26
        self.ema_hs_ms_per_block = 0.66
        self.base_prefill_launch_ms = 42.0
        self.base_restore_rpc_ms = 12.0

        self._active_reads: dict[str, int] = {}
        self._active_saves: dict[str, int] = {}
        self.decisions_restore = 0
        self.decisions_bypass_cost = 0
        self.decisions_truncated_budget = 0

    @property
    def shadow_hs_bytes(self) -> int:
        return sum(self._active_reads.values()) + sum(self._active_saves.values())

    @property
    def active_read_count(self) -> int:
        return len(self._active_reads)

    @property
    def active_save_count(self) -> int:
        return len(self._active_saves)

    def update_ema(self, kv_ms_samples: Sequence[float], hs_ms_samples: Sequence[float]) -> None:
        alpha = 0.25
        for val in kv_ms_samples:
            if 0.02 <= val <= 10.0:
                self.ema_kv_ms_per_block = (1.0 - alpha) * self.ema_kv_ms_per_block + alpha * val
        for val in hs_ms_samples:
            if 0.02 <= val <= 10.0:
                self.ema_hs_ms_per_block = (1.0 - alpha) * self.ema_hs_ms_per_block + alpha * val

    def reserve_read(self, req_id: str, hs_tokens: int) -> None:
        reserve = (3 * hs_tokens * self.row_bytes + self.header_allowance) if hs_tokens > 0 else 0
        self._active_reads[req_id] = reserve

    def release_read(self, req_id: str) -> None:
        self._active_reads.pop(req_id, None)

    def reserve_save(self, req_id: str, hs_tokens: int) -> None:
        if hs_tokens > 0:
            self._active_saves[req_id] = 3 * hs_tokens * self.row_bytes + self.header_allowance

    def release_save(self, req_id: str) -> None:
        self._active_saves.pop(req_id, None)

    def available_hs_save_blocks(self) -> int:
        free_bytes = max(0, self.max_hs_bytes - self.shadow_hs_bytes - self.header_allowance)
        per_block_reserve = 3 * self.block_size * self.row_bytes
        return free_bytes // per_block_reserve if per_block_reserve > 0 else 0

    def estimate_prefill_saved_ms(
        self, start_tokens: int, end_tokens: int, active_gpu_prefills: int = 0
    ) -> float:
        start_blk = start_tokens // self.block_size
        end_blk = end_tokens // self.block_size
        if end_blk <= start_blk:
            return 0.0
        compute_ms = sum(marginal_prefill_ms_per_block(b) for b in range(start_blk, end_blk))
        if start_blk == 0:
            compute_ms += self.base_prefill_launch_ms * min(1.0, (end_blk - start_blk) / 28.0)
        return compute_ms * (1.0 + 0.32 * max(0, active_gpu_prefills))

    def estimate_restore_ms(
        self,
        ext_kv_blocks: int,
        ext_hs_blocks: int,
    ) -> float:
        if ext_kv_blocks <= 0 and ext_hs_blocks <= 0:
            return 0.0
        q_reads = self.active_read_count
        q_saves = self.active_save_count
        kv_part = ext_kv_blocks * self.ema_kv_ms_per_block * (1.0 + 0.65 * q_reads + 0.25 * q_saves)
        hs_part = ext_hs_blocks * self.ema_hs_ms_per_block * (1.0 + 0.65 * q_reads + 0.30 * q_saves)
        return self.base_restore_rpc_ms + kv_part + hs_part

    def select_external_tokens(
        self,
        *,
        local_tokens: int,
        candidate_total_tokens: int,
        is_text_only: bool,
        host_hs_covers_local: bool,
        enable_cost_gate: bool,
        active_gpu_prefills: int = 0,
    ) -> int:
        """Return optimal total cached token target ``>= local_tokens``."""
        ext_tokens = candidate_total_tokens - local_tokens
        if ext_tokens <= 0:
            return local_tokens

        if self.active_read_count >= self.max_pending_transfers:
            self.decisions_truncated_budget += 1
            return local_tokens

        # 1. Shadow HS memory budget pre-check (eliminates late start_load_kv failure)
        target_total = candidate_total_tokens
        if not is_text_only:
            hs_tokens_needed = (
                (target_total - local_tokens) if host_hs_covers_local else target_total
            )
            free_bytes = max(0, self.max_hs_bytes - self.shadow_hs_bytes - self.header_allowance)
            max_hs_fit_blocks = free_bytes // (3 * self.block_size * self.row_bytes)
            max_hs_fit_tokens = max_hs_fit_blocks * self.block_size
            if hs_tokens_needed > max_hs_fit_tokens:
                self.decisions_truncated_budget += 1
                if host_hs_covers_local:
                    target_total = local_tokens + max_hs_fit_tokens
                else:
                    target_total = max_hs_fit_tokens
                if target_total <= local_tokens:
                    return local_tokens

        if not enable_cost_gate:
            self.decisions_restore += 1
            return target_total

        # 2. Compare estimated external restore latency against MoE GPU prefill savings
        ext_kv_blocks = (target_total - local_tokens) // self.block_size
        if is_text_only:
            ext_hs_blocks = 0
        elif host_hs_covers_local:
            ext_hs_blocks = ext_kv_blocks
        else:
            ext_hs_blocks = target_total // self.block_size

        saved_ms = self.estimate_prefill_saved_ms(local_tokens, target_total, active_gpu_prefills)
        restore_ms = self.estimate_restore_ms(ext_kv_blocks, ext_hs_blocks)

        if restore_ms < saved_ms:
            self.decisions_restore += 1
            return target_total

        self.decisions_bypass_cost += 1
        return local_tokens


class UnifiedPrefixDirectory:
    """Central multi-tier prefix directory & dependency-consistent retention engine."""

    def __init__(
        self,
        *,
        block_size: int,
        kv_block_bytes: int,
        hs_block_bytes: int,
        settings: MooncakeConfig,
        max_sketch_entries: int = 65536,
    ) -> None:
        self.block_size = block_size
        self.kv_block_bytes = kv_block_bytes
        self.hs_block_bytes = hs_block_bytes
        self.settings = settings
        self.max_sketch_entries = max_sketch_entries

        self.step_seq = 0
        self._roots: dict[bytes, PrefixNode] = {}
        self._nodes: dict[bytes, PrefixNode] = {}
        self._sketch: OrderedDict[bytes, int] = OrderedDict()
        self._sketch_audio: set[bytes] = set()
        self._visited_reqs: OrderedDict[str, bool] = OrderedDict()
        self._shadow_host_hs: OrderedDict[bytes, int] = OrderedDict()
        self._shadow_host_hs_bytes = 0
        self.store_bytes_used = 0

        # Telemetry counters for Chapter 3 evaluation
        self.cold_filtered_requests = 0
        self.admitted_write_requests = 0
        self.evicted_blocks_count = 0
        self.evicted_bytes_total = 0

    def _get_or_create_chain(self, block_hashes: Sequence[bytes]) -> list[PrefixNode]:
        chain: list[PrefixNode] = []
        parent: PrefixNode | None = None
        for depth, raw_hash in enumerate(block_hashes):
            key = bytes(raw_hash)
            node = self._nodes.get(key)
            if node is None:
                node = PrefixNode(block_hash=key, depth=depth, parent=parent)
                self._nodes[key] = node
                if parent is None:
                    self._roots[key] = node
                else:
                    parent.children[key] = node
            chain.append(node)
            parent = node
        return chain

    def record_request_visit(
        self, req_id: str, block_hashes: Sequence[bytes], *, is_text_only: bool
    ) -> list[int]:
        """Idempotently record a request's prefix visit; return visit counts per block."""
        already_counted = req_id in self._visited_reqs
        if not already_counted:
            self._visited_reqs[req_id] = True
            if len(self._visited_reqs) > 16384:
                self._visited_reqs.popitem(last=False)
            self.step_seq += 1

        counts: list[int] = []
        for raw_hash in block_hashes:
            key = bytes(raw_hash)
            if not already_counted:
                prev = self._sketch.pop(key, 0) + 1
                self._sketch[key] = prev
                if not is_text_only:
                    self._sketch_audio.add(key)
                if len(self._sketch) > self.max_sketch_entries:
                    dropped, _ = self._sketch.popitem(last=False)
                    self._sketch_audio.discard(dropped)
                if (node := self._nodes.get(key)) is not None:
                    node.access_count += 1
                    if not is_text_only:
                        node.audio_access_count += 1
                    node.last_step = self.step_seq
            counts.append(self._sketch.get(key, 1))
        return counts

    def retire_request(self, req_id: str) -> None:
        self._visited_reqs.pop(req_id, None)

    def record_shadow_host_hs(self, block_hashes: Sequence[bytes]) -> None:
        """Deterministically mirror Worker's HostHSCache LRU on Scheduler."""
        if not self.settings.enable_local_tier:
            return
        budget = self.settings.host_hs_cache_bytes
        bsize = self.hs_block_bytes
        if bsize <= 0 or bsize > budget:
            return
        chain = self._get_or_create_chain(block_hashes)
        for node in chain:
            key = node.block_hash
            if key in self._shadow_host_hs:
                self._shadow_host_hs.move_to_end(key)
            else:
                while self._shadow_host_hs_bytes + bsize > budget and self._shadow_host_hs:
                    ev_key, ev_b = self._shadow_host_hs.popitem(last=False)
                    self._shadow_host_hs_bytes -= ev_b
                    if (ev_node := self._nodes.get(ev_key)) is not None:
                        ev_node.host_hs = False
                self._shadow_host_hs[key] = bsize
                self._shadow_host_hs_bytes += bsize
            node.host_hs = True

    def reconcile_worker_meta(self, meta: CoStateWorkerMetadata) -> None:
        for raw in meta.host_hs_added:
            key = bytes(raw)
            if (node := self._nodes.get(key)) is not None:
                node.host_hs = True
        for raw in meta.host_hs_evicted:
            key = bytes(raw)
            if (ev := self._shadow_host_hs.pop(key, None)) is not None:
                self._shadow_host_hs_bytes -= ev
            if (node := self._nodes.get(key)) is not None:
                node.host_hs = False
        for raw in meta.store_kv_saved:
            key = bytes(raw)
            if (node := self._nodes.get(key)) is not None and not node.store_kv:
                node.store_kv = True
                self.store_bytes_used += self.kv_block_bytes
        for raw in meta.store_hs_saved:
            key = bytes(raw)
            if (node := self._nodes.get(key)) is not None and not node.store_hs:
                node.store_hs = True
                self.store_bytes_used += self.hs_block_bytes

    def contiguous_hs_tokens(self, block_hashes: Sequence[bytes]) -> int:
        """Longest contiguous prefix whose HS is available in HostHSCache or Store."""
        matched = 0
        for raw_hash in block_hashes:
            node = self._nodes.get(bytes(raw_hash))
            if node is None or not (node.host_hs or node.store_hs):
                break
            matched += self.block_size
        return matched

    def host_hs_covers_prefix(self, block_hashes: Sequence[bytes], num_tokens: int) -> bool:
        blocks = num_tokens // self.block_size
        if blocks <= 0:
            return True
        if blocks > len(block_hashes):
            return False
        for raw_hash in block_hashes[:blocks]:
            node = self._nodes.get(bytes(raw_hash))
            if node is None or not node.host_hs:
                return False
        return True

    def observe_lookup(
        self, block_hashes: Sequence[bytes], kv_hit_tokens: int, joint_hit_tokens: int
    ) -> None:
        """Synchronize directory with authoritative Store lookup results."""
        kv_blocks = kv_hit_tokens // self.block_size
        joint_blocks = joint_hit_tokens // self.block_size
        max_blocks = max(kv_blocks, joint_blocks)
        if max_blocks <= 0:
            return
        chain = self._get_or_create_chain(block_hashes[:max_blocks])
        for idx, node in enumerate(chain):
            if idx < kv_blocks and not node.store_kv:
                node.store_kv = True
                self.store_bytes_used += self.kv_block_bytes
            if idx < joint_blocks and not node.store_hs:
                node.store_hs = True
                self.store_bytes_used += self.hs_block_bytes

    def _node_store_bytes(self, node: PrefixNode) -> int:
        total = 0
        if node.store_kv:
            total += self.kv_block_bytes
        if node.store_hs:
            total += self.hs_block_bytes
        return total

    def _is_store_leaf(self, node: PrefixNode) -> bool:
        return node.in_store and not any(child.in_store for child in node.children.values())

    def _leaf_utility_score(self, node: PrefixNode) -> float:
        bbytes = max(1, self._node_store_bytes(node))
        age = max(1, self.step_seq - node.last_step + 1)
        recency_factor = 1.0 / (1.0 + 0.05 * age)
        freq = max(1, node.access_count)
        # Shared prefix protection: historical branching + audio dual-modality utility
        modal_bonus = 1.15 if node.audio_access_count > 0 else 1.0
        save_ms = marginal_prefill_ms_per_block(node.depth)
        return (freq * recency_factor * modal_bonus * save_ms * 1e6) / bbytes

    def _select_eviction_victim(
        self, policy: str, protected_hashes: set[bytes]
    ) -> PrefixNode | None:
        if policy == "coupled_lru":
            candidates = [
                n
                for n in self._nodes.values()
                if n.in_store and n.active_refs == 0 and n.block_hash not in protected_hashes
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda n: (n.last_step, -n.depth))

        # Both "topo_lru" and "costate_utility" strictly enforce leaf-to-root topology!
        leaves = [
            n
            for n in self._nodes.values()
            if self._is_store_leaf(n) and n.active_refs == 0 and n.block_hash not in protected_hashes
        ]
        if not leaves:
            return None
        if policy == "topo_lru":
            return min(leaves, key=lambda n: (n.last_step, -n.depth))
        return min(leaves, key=lambda n: (self._leaf_utility_score(n), n.last_step))

    def plan_store_write_and_evictions(
        self,
        *,
        block_hashes: Sequence[bytes],
        target_tokens: int,
        is_text_only: bool,
        avail_hs_blocks: int,
    ) -> tuple[int, int, list[bytes]]:
        """Decide ``(save_kv_tokens, save_hs_tokens, evict_hashes)`` for Store."""
        num_blocks = min(len(block_hashes), target_tokens // self.block_size)
        if num_blocks <= 0:
            return 0, 0, []

        # 1. Mechanism 3A: N-Hit write admission filter (eliminates one-off cold pollution)
        min_hits = self.settings.min_write_hits
        eligible_blocks = 0
        for raw_hash in block_hashes[:num_blocks]:
            if self._sketch.get(bytes(raw_hash), 0) < min_hits:
                break
            eligible_blocks += 1

        if eligible_blocks == 0:
            self.cold_filtered_requests += 1
            return 0, 0, []

        chain = self._get_or_create_chain(block_hashes[:eligible_blocks])
        want_hs = not (is_text_only and self.settings.enable_modality_decouple)

        # 2. Mechanism 3C: Synchronous (KV, HS) write truncation when HS staging is bounded
        if want_hs and self.settings.sync_write_truncation:
            unsaved_hs = [i for i, node in enumerate(chain) if not node.store_hs]
            if len(unsaved_hs) > avail_hs_blocks:
                cutoff_idx = unsaved_hs[avail_hs_blocks]
                chain = chain[:cutoff_idx]
                if not chain:
                    return 0, 0, []

        # 3. Mechanism 3B: Managed Store quota & bottom-up leaf-to-root eviction
        evicted_hashes: list[bytes] = []
        max_store = self.settings.max_store_bytes
        policy = self.settings.eviction_policy
        protected = {node.block_hash for node in chain}

        accepted_blocks = 0
        for node in chain:
            add_kv = not node.store_kv
            add_hs = want_hs and not node.store_hs
            need_bytes = (self.kv_block_bytes if add_kv else 0) + (self.hs_block_bytes if add_hs else 0)

            if max_store > 0 and need_bytes > 0:
                if policy == "none":
                    if self.store_bytes_used + need_bytes > max_store:
                        break
                else:
                    # Only evict victims strictly less valuable than new block when utility-driven
                    new_score = (
                        max(1, self._sketch.get(node.block_hash, 1))
                        * (1.15 if node.block_hash in self._sketch_audio else 1.0)
                        * marginal_prefill_ms_per_block(node.depth)
                        * 1e6
                        / max(1, need_bytes)
                    )
                    while self.store_bytes_used + need_bytes > max_store:
                        victim = self._select_eviction_victim(policy, protected)
                        if victim is None:
                            break
                        if policy == "costate_utility" and self._leaf_utility_score(victim) > new_score * 1.05:
                            victim = None
                            break
                        freed = self._node_store_bytes(victim)
                        victim.store_kv = False
                        victim.store_hs = False
                        self.store_bytes_used = max(0, self.store_bytes_used - freed)
                        self.evicted_blocks_count += 1
                        self.evicted_bytes_total += freed
                        evicted_hashes.append(victim.block_hash)
                    if self.store_bytes_used + need_bytes > max_store:
                        break

            if add_kv:
                node.store_kv = True
                self.store_bytes_used += self.kv_block_bytes
            if add_hs:
                node.store_hs = True
                self.store_bytes_used += self.hs_block_bytes
            node.last_step = self.step_seq
            accepted_blocks += 1

        if accepted_blocks > 0:
            self.admitted_write_requests += 1
        save_kv_tokens = accepted_blocks * self.block_size
        save_hs_tokens = save_kv_tokens if want_hs else 0
        return save_kv_tokens, save_hs_tokens, evicted_hashes
