# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chapter 3 Comprehensive Evaluation Suite for CoState on Qwen3-Omni-30B-A3B.

Supports three reproducible evaluation suites in a single self-terminating run:
1. ``exactness``: Verifies bit/token-exact output parity across Text-only KV
   reuse, Local GPU KV + Host HS tiered reuse, and External Store reuse.
2. ``read_matrix``: Controlled evaluation of Mechanism 1 (Modality & Tier-Aware
   On-Demand Recovery) and Mechanism 2 (Online Cost-Aware Recovery Gate) across
   512 / 2048 / 4096 / 8192 token prompts, Text-only vs Audio, and c=1, 2.
3. ``e2e_serving``: Realistic 60-request multi-turn shared-prefix + one-off
   cold-noise mixed omni-modal serving trace under constrained GPU & Store
   quotas, measuring TTFT, TTFA, Throughput, SLO Goodput, and Store ZBR.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

import psutil
import pynvml
import torch

from tests.engine import kv_offload_helpers as helpers
from vllm_omni.worker.gpu_ar_worker import GPUARWorker

REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
MODES = ("off", "pr10107", "pr10107_tuned", "topo_lru", "costate_ext", "costate")


class CoStateEvalWorker(GPUARWorker):
    """Non-intrusive worker telemetry & direct Mooncake Store ZBR auditor."""

    def init_device(self):
        from vllm_omni.distributed.kv_transfer.mooncake.connector import (
            JointStoreWorker,
            OmniMooncakeStoreConnector,
        )

        self._diag_lock = threading.Lock()
        self._connector_ref = None
        self._observed_chains: set[tuple[bytes, ...]] = set()
        self._reset_diag_buffers()

        if not getattr(JointStoreWorker.lookup, "_costate_eval_wrapped", False):
            orig_lookup = JointStoreWorker.lookup

            def timed_lookup(worker_self, num_tokens, block_hashes):
                t0 = time.perf_counter()
                res = orig_lookup(worker_self, num_tokens, block_hashes)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                owner = getattr(worker_self, "_eval_owner", None)
                if owner is not None:
                    with owner._diag_lock:
                        owner._diag["lookup_calls"] += 1
                        owner._diag["lookup_ms"].append(dt_ms)
                return res

            timed_lookup._costate_eval_wrapped = True  # type: ignore[attr-defined]
            JointStoreWorker.lookup = timed_lookup

        if not getattr(OmniMooncakeStoreConnector.bind_connector_metadata, "_costate_eval_wrapped", False):
            orig_bind = OmniMooncakeStoreConnector.bind_connector_metadata

            def tracked_bind(conn_self, connector_metadata):
                orig_bind(conn_self, connector_metadata)
                owner = getattr(conn_self, "_eval_owner", None)
                if owner is not None:
                    with conn_self._lock:
                        chains = [
                            tuple(bytes(h) for h in a.hashes[: max(0, a.prompt_len // conn_self.block_size)])
                            for a in conn_self._attempts.values()
                            if a.hashes and a.prompt_len >= conn_self.block_size
                        ]
                    if chains:
                        with owner._diag_lock:
                            for ch in chains:
                                if ch:
                                    owner._observed_chains.add(ch)

            tracked_bind._costate_eval_wrapped = True  # type: ignore[attr-defined]
            OmniMooncakeStoreConnector.bind_connector_metadata = tracked_bind

        if not getattr(OmniMooncakeStoreConnector._read_hs_range, "_costate_eval_wrapped", False):
            orig_read_range = OmniMooncakeStoreConnector._read_hs_range

            def timed_read_range(conn_self, attempt, start_tokens, end_tokens, prefix_tensors=None):
                t0 = time.perf_counter()
                out = orig_read_range(conn_self, attempt, start_tokens, end_tokens, prefix_tensors)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                owner = getattr(conn_self, "_eval_owner", None)
                if owner is not None:
                    with owner._diag_lock:
                        owner._diag["ext_hs_reads"] += 1
                        owner._diag["ext_hs_read_ms"].append(dt_ms)
                        owner._diag["ext_hs_read_tokens"].append(max(0, end_tokens - start_tokens))
                return out

            timed_read_range._costate_eval_wrapped = True  # type: ignore[attr-defined]
            OmniMooncakeStoreConnector._read_hs_range = timed_read_range

        if not getattr(OmniMooncakeStoreConnector._save_hs, "_costate_eval_wrapped", False):
            orig_save_hs = OmniMooncakeStoreConnector._save_hs

            def timed_save_hs(conn_self, save):
                t0 = time.perf_counter()
                nblk = len(save.blocks)
                ok = orig_save_hs(conn_self, save)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                owner = getattr(conn_self, "_eval_owner", None)
                if owner is not None:
                    with owner._diag_lock:
                        owner._diag["save_batches"] += 1
                        owner._diag["save_blocks_total"] += nblk
                        owner._diag["save_hs_ms"].append(dt_ms)
                        if not ok:
                            owner._diag["save_failures"] += 1
                return ok

            timed_save_hs._costate_eval_wrapped = True  # type: ignore[attr-defined]
            OmniMooncakeStoreConnector._save_hs = timed_save_hs

        if not getattr(OmniMooncakeStoreConnector.capture_step, "_costate_eval_wrapped", False):
            orig_capture = OmniMooncakeStoreConnector.capture_step

            def timed_capture(conn_self, req_id, attempt_id, start_token, prompt_len, tensors):
                t0 = time.perf_counter()
                merged = orig_capture(conn_self, req_id, attempt_id, start_token, prompt_len, tensors)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                owner = getattr(conn_self, "_eval_owner", None)
                if owner is not None:
                    with owner._diag_lock:
                        owner._diag["capture_calls"] += 1
                        owner._diag["capture_ms"].append(dt_ms)
                        if merged is not None:
                            inserted = merged["embedding"].shape[0] - tensors["embedding"].shape[0]
                            if inserted > 0:
                                owner._diag["restored_hs_rows"] += inserted
                return merged

            timed_capture._costate_eval_wrapped = True  # type: ignore[attr-defined]
            OmniMooncakeStoreConnector.capture_step = timed_capture

        super().init_device()
        runner_kc = getattr(getattr(self, "model_runner", None), "kv_connector", None)
        inner_conn = getattr(runner_kc, "kv_connector", runner_kc)
        if inner_conn is not None and hasattr(inner_conn, "audit_store_prefix_chains"):
            self._connector_ref = inner_conn
            inner_conn._eval_owner = self
            if getattr(inner_conn, "connector_worker", None) is not None:
                inner_conn.connector_worker._eval_owner = self

    def _reset_diag_buffers(self):
        self._diag = {
            "lookup_calls": 0,
            "lookup_ms": [],
            "ext_hs_reads": 0,
            "ext_hs_read_ms": [],
            "ext_hs_read_tokens": [],
            "save_batches": 0,
            "save_blocks_total": 0,
            "save_hs_ms": [],
            "save_failures": 0,
            "capture_calls": 0,
            "capture_ms": [],
            "restored_hs_rows": 0,
        }

    def costate_eval_snapshot(self, reset_chains: bool = False, include_audit: bool = False):
        runner_kc = getattr(getattr(self, "model_runner", None), "kv_connector", None)
        inner_conn = self._connector_ref or getattr(runner_kc, "kv_connector", runner_kc)
        if inner_conn is not None and getattr(inner_conn, "_eval_owner", None) is None:
            self._connector_ref = inner_conn
            inner_conn._eval_owner = self
            if getattr(inner_conn, "connector_worker", None) is not None:
                inner_conn.connector_worker._eval_owner = self

        with self._diag_lock:
            snap = dict(self._diag)
            chains = list(self._observed_chains)
            self._reset_diag_buffers()
            if reset_chains:
                self._observed_chains.clear()

        audit = {}
        if include_audit and inner_conn is not None and hasattr(inner_conn, "audit_store_prefix_chains"):
            audit = inner_conn.audit_store_prefix_chains(chains)

        proc = psutil.Process()
        return {
            "pid": proc.pid,
            "rss_bytes": proc.memory_info().rss,
            "diag": snap,
            "store_audit": audit,
            "observed_chain_count": len(chains),
        }


def build_scaled_body(tag: str, target_tokens: int) -> str:
    base_facts = helpers._FACTS
    needed_facts = max(1, int(target_tokens / 26.0))
    pieces = []
    for idx in range(needed_facts):
        fact = base_facts[idx % len(base_facts)]
        pieces.append(f"[{tag}-S{idx // len(base_facts)}-F{idx}] {fact}")
    return " ".join(pieces)


def build_prompt_from_segments(segments: list[tuple[str, int]], question: str) -> dict:
    parts = [build_scaled_body(seg_tag, seg_tokens) for seg_tag, seg_tokens in segments]
    joined = " ".join(parts)
    text = f"<|im_start|>user\n{joined}\n{question}<|im_end|>\n<|im_start|>assistant\n"
    return {"prompt": text}


def build_scaled_prompt(document: str, target_tokens: int, turn_suffix: str = "State Fact 0 briefly.") -> dict:
    return build_prompt_from_segments([(document, target_tokens)], turn_suffix)


def stats_dict(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p95": ordered[p95_idx],
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "max": ordered[-1],
    }


def deployment(
    *,
    mode: str,
    store_config: str | None,
    namespace: str,
    max_concurrency: int,
    kv_bytes: int,
    max_model_len: int,
    max_store_bytes: int,
    host_hs_bytes: int,
    force_min_write_hits: int | None = None,
) -> dict:
    enable_local = mode == "costate"
    thinker: dict = {
        "use_v2_model_runner": True,
        "worker_cls": "benchmarks.qwen3_omni.costate_ch3_eval.CoStateEvalWorker",
        "tensor_parallel_size": 1,
        "max_num_seqs": max_concurrency,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": min(2048, max(256, max_model_len // 2)),
        "enable_chunked_prefill": True,
        "enable_prefix_caching": enable_local,
        "block_size": 16,
        "gpu_memory_utilization": 0.88,
        "kv_cache_memory_bytes": kv_bytes,
        "revision": REVISION,
    }
    if mode != "off":
        assert store_config is not None, "Mooncake store_config required for cached modes"
        mc_cfg: dict[str, object] = {
            "config_file": str(Path(store_config).resolve()),
            "namespace": namespace,
            "model_revision": REVISION,
            "max_pending_transfers": 16,
        }
        if mode == "pr10107":
            mc_cfg["max_hs_bytes"] = 256 * 1024 * 1024
            mc_cfg["enable_costate"] = False
        elif mode == "pr10107_tuned":
            mc_cfg.update(
                max_hs_bytes=1024 * 1024 * 1024,
                enable_costate=True,
                enable_local_tier=False,
                enable_modality_decouple=False,
                enable_cost_gate=False,
                min_write_hits=force_min_write_hits or 1,
                eviction_policy="coupled_lru",
                sync_write_truncation=True,
                max_store_bytes=max_store_bytes,
                host_hs_cache_bytes=host_hs_bytes,
            )
        elif mode == "topo_lru":
            mc_cfg.update(
                max_hs_bytes=1024 * 1024 * 1024,
                enable_costate=True,
                enable_local_tier=False,
                enable_modality_decouple=False,
                enable_cost_gate=False,
                min_write_hits=force_min_write_hits or 1,
                eviction_policy="topo_lru",
                sync_write_truncation=True,
                max_store_bytes=max_store_bytes,
                host_hs_cache_bytes=host_hs_bytes,
            )
        elif mode == "costate_ext":
            mc_cfg.update(
                max_hs_bytes=1024 * 1024 * 1024,
                enable_costate=True,
                enable_local_tier=False,
                enable_modality_decouple=True,
                enable_cost_gate=True,
                min_write_hits=force_min_write_hits or 2,
                eviction_policy="costate_utility",
                sync_write_truncation=True,
                max_store_bytes=max_store_bytes,
                host_hs_cache_bytes=host_hs_bytes,
            )
        elif mode == "costate":
            mc_cfg.update(
                max_hs_bytes=1024 * 1024 * 1024,
                enable_costate=True,
                enable_local_tier=True,
                enable_modality_decouple=True,
                enable_cost_gate=True,
                min_write_hits=force_min_write_hits or 2,
                eviction_policy="costate_utility",
                sync_write_truncation=True,
                max_store_bytes=max_store_bytes,
                host_hs_cache_bytes=host_hs_bytes,
            )
        thinker["omni_kv_config"] = {"kv_store_config": {"mooncake_config": mc_cfg}}

    return helpers.stage_overrides(
        lmcache=False,
        thinker_extra=thinker,
        downstream_extra={
            "1": {
                "max_num_seqs": max_concurrency,
                "gpu_memory_utilization": 0.35,
                "revision": REVISION,
                "subtalker_sampling_params": {"do_sample": False},
            },
            "2": {
                "max_num_seqs": max_concurrency,
                "gpu_memory_utilization": 0.20,
                "revision": REVISION,
            },
        },
    )


async def run_request(
    omni,
    item: dict,
    request_id: str,
    *,
    want_audio: bool,
    max_tokens: int = 24,
    kind: str = "default",
) -> dict:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    greedy = {**helpers.GREEDY, "max_tokens": max_tokens}
    params = [SamplingParams(**greedy, output_kind=RequestOutputKind.DELTA) for _ in range(omni.num_stages)]
    modalities = ["text", "audio"] if want_audio else ["text"]
    start = time.perf_counter()
    res = {
        "request_id": request_id,
        "kind": kind,
        "modality": "audio" if want_audio else "text",
        "ttft_ms": None,
        "first_audio_ms": None,
        "cached_tokens": 0,
        "prompt_tokens": 0,
    }
    tokens, audio, text_parts = [], [], []
    async for out in omni.generate(
        item,
        request_id=request_id,
        sampling_params_list=params,
        output_modalities=modalities,
    ):
        if out.error:
            raise RuntimeError(out.error)
        elapsed = (time.perf_counter() - start) * 1000.0
        if out.stage_id == 0:
            res["cached_tokens"] = max(res["cached_tokens"], out.num_cached_tokens or 0)
            res["prompt_tokens"] = len(out.prompt_token_ids or []) or res["prompt_tokens"]
            for completion in out.outputs:
                if completion.token_ids:
                    if res["ttft_ms"] is None:
                        res["ttft_ms"] = elapsed
                    tokens.extend(completion.token_ids)
                    text_parts.append(completion.text)
        if want_audio and out.final_output_type == "audio":
            value = out.multimodal_output.get("audio")
            if isinstance(value, torch.Tensor) and value.numel():
                if res["first_audio_ms"] is None:
                    res["first_audio_ms"] = elapsed
                audio.append(value)
    res["e2e_ms"] = (time.perf_counter() - start) * 1000.0
    res["text"] = "".join(text_parts)
    res["token_ids"] = tokens
    res["num_gen_tokens"] = len(tokens)
    waveform = torch.cat([piece.detach().cpu().float().reshape(-1) for piece in audio]) if audio else torch.empty(0)
    res["audio_samples"] = waveform.numel()
    res["audio_sha256"] = hashlib.sha256(waveform.numpy().tobytes()).hexdigest() if waveform.numel() else ""
    res["success"] = bool(tokens and (waveform.numel() if want_audio else True))
    return res


async def fetch_worker_snapshot(omni, *, reset_chains: bool = False, include_audit: bool = False) -> dict:
    snaps = await omni.collective_rpc(
        "costate_eval_snapshot",
        args=(reset_chains, include_audit),
        stage_ids=[0],
    )
    return next(
        (
            item
            for entry in (snaps or [])
            for item in (entry if isinstance(entry, list) else [entry])
            if isinstance(item, dict) and "diag" in item
        ),
        {},
    )


def build_e2e_serving_trace(seed: int = 42) -> list[dict]:
    """Build a deterministic 60-request mixed multi-turn + cold-noise omni workload.

    Working set structure:
    - 2 shared domain system roots (~1024 tokens each)
    - 8 session branches extending the roots to ~2048 and ~3584 tokens across turns
      (Zipf-skewed so hot sessions repeat 5-7x while cold branches repeat 2x)
    - 25% (15/60) one-off cold noise documents (~4096 and ~8192 tokens) that never
      repeat, testing cold-write overhead, 2-Hit admission, and zombie-block safety
    - 60% Audio-output turns + 40% Text-only turns sharing prefixes
    """
    rng = random.Random(seed)
    # 8 session specs: (root_id, branch_id, branch_extra_toks, weight)
    families = [
        ("sysA", "sess0", 1024, 7),
        ("sysA", "sess1", 1024, 6),
        ("sysA", "sess2", 2560, 5),
        ("sysB", "sess3", 1024, 6),
        ("sysB", "sess4", 2560, 5),
        ("sysB", "sess5", 1024, 3),
        ("sysA", "sess6", 2560, 2),
        ("sysB", "sess7", 2560, 2),
    ]
    hot_pool: list[tuple[str, str, int, int]] = []
    turn_counters = {f[1]: 0 for f in families}
    for root_id, branch_id, extra_toks, count in families:
        for _ in range(count):
            hot_pool.append((root_id, branch_id, extra_toks, 0))
    rng.shuffle(hot_pool)
    hot_pool = hot_pool[:45]

    noise_specs = [
        ("noise4k", 4096) if i % 3 != 0 else ("noise8k", 8192)
        for i in range(15)
    ]

    # Interleave 45 multi-turn family requests and 15 one-off cold noise requests into 60 slots
    trace: list[dict] = []
    h_idx, n_idx = 0, 0
    for slot in range(60):
        if slot % 4 == 2 and n_idx < len(noise_specs):
            ntag, nlen = noise_specs[n_idx]
            n_idx += 1
            want_audio = (slot % 5 != 0)
            prompt = build_scaled_prompt(
                f"oneoff-{ntag}-{n_idx}",
                nlen,
                turn_suffix=f"Summarize item {n_idx % 8} in one sentence.",
            )
            trace.append(
                {
                    "slot": slot,
                    "kind": "cold_noise",
                    "want_audio": want_audio,
                    "target_len": nlen,
                    "item": prompt,
                }
            )
        else:
            root_id, branch_id, extra_toks, _ = hot_pool[h_idx]
            h_idx += 1
            tnum = turn_counters[branch_id]
            turn_counters[branch_id] += 1
            # Alternate Audio (60%) and Text-only (40%) across turns of the SAME shared session!
            want_audio = (h_idx % 5) not in (1, 3)
            segments = [(f"root-{root_id}", 1024), (f"branch-{branch_id}", extra_toks)]
            question = f"Turn {tnum}: state fact {tnum % 6} from the context in one concise sentence."
            prompt = build_prompt_from_segments(segments, question)
            trace.append(
                {
                    "slot": slot,
                    "kind": "hot_audio" if want_audio else "hot_text",
                    "family": branch_id,
                    "turn": tnum,
                    "want_audio": want_audio,
                    "target_len": 1024 + extra_toks,
                    "item": prompt,
                }
            )
    return trace


async def run_exactness_suite(omni, report: dict) -> None:
    cases = []
    for length in (512, 2048):
        base_doc = f"exact-doc-{length}"
        # Pass 0 (cold audio), Pass 1 (hit1 text-only), Pass 2 (hit2 audio -> local/store hit)
        for pass_idx, want_audio in ((0, True), (1, False), (2, True)):
            prompt = build_scaled_prompt(base_doc, length, "Repeat Fact 0 word for word.")
            res = await run_request(
                omni,
                prompt,
                f"exact-l{length}-p{pass_idx}",
                want_audio=want_audio,
                max_tokens=24,
                kind=f"pass{pass_idx}_{'audio' if want_audio else 'text'}",
            )
            await asyncio.sleep(0.4)
            cases.append({"length": length, "pass": pass_idx, "want_audio": want_audio, "result": res})
    report["exactness_cases"] = cases


async def run_read_matrix_suite(omni, args, report: dict) -> None:
    waves = []
    for target_len in args.prompt_lengths:
        for concurrency in args.concurrency:
            docs = [f"rmat-l{target_len}-c{concurrency}-d{i}" for i in range(concurrency)]
            prompts = [build_scaled_prompt(d, target_len) for d in docs]

            # Warmup/populate prefix (2 passes when mode uses 2-Hit Store admission so Store is populated!)
            populate_passes = 2 if args.mode in ("costate_ext", "costate") else 1
            for p_idx in range(populate_passes):
                await asyncio.gather(
                    *[
                        run_request(
                            omni,
                            prompts[i],
                            f"pop-l{target_len}-c{concurrency}-p{p_idx}-i{i}",
                            want_audio=True,
                            max_tokens=8,
                            kind="populate",
                        )
                        for i in range(concurrency)
                    ]
                )
                await asyncio.sleep(1.0)
            await fetch_worker_snapshot(omni, reset_chains=False, include_audit=False)

            for want_audio in (False, True):
                mod_name = "audio" if want_audio else "text"
                for rep in range(args.repeats):
                    t0 = time.perf_counter()
                    reqs = await asyncio.gather(
                        *[
                            run_request(
                                omni,
                                prompts[i],
                                f"hit-{mod_name}-l{target_len}-c{concurrency}-r{rep}-i{i}",
                                want_audio=want_audio,
                                max_tokens=24,
                                kind=f"hit_{mod_name}",
                            )
                            for i in range(concurrency)
                        ]
                    )
                    wall_s = time.perf_counter() - t0
                    await asyncio.sleep(0.15)
                    snap = await fetch_worker_snapshot(omni, reset_chains=False, include_audit=False)
                    row = {
                        "length": target_len,
                        "concurrency": concurrency,
                        "modality": mod_name,
                        "repeat": rep,
                        "wall_s": wall_s,
                        "ttft_stats": stats_dict([r["ttft_ms"] for r in reqs if r["ttft_ms"] is not None]),
                        "ttfa_stats": stats_dict(
                            [r["first_audio_ms"] for r in reqs if r["first_audio_ms"] is not None]
                        ),
                        "e2e_stats": stats_dict([r["e2e_ms"] for r in reqs]),
                        "cached_tokens": [r["cached_tokens"] for r in reqs],
                        "prompt_tokens": [r["prompt_tokens"] for r in reqs],
                        "diag": snap.get("diag", {}),
                    }
                    waves.append(row)
                    print(
                        f"[read_matrix|{args.mode}] len={target_len} c={concurrency} mod={mod_name} r={rep} "
                        f"cached={row['cached_tokens']} TTFT={row['ttft_stats']['mean']:.1f}ms "
                        f"TTFA={row['ttfa_stats']['mean'] or 0.0:.1f}ms",
                        flush=True,
                    )
    report["read_matrix"] = waves


async def run_e2e_serving_suite(omni, args, report: dict) -> None:
    trace = build_e2e_serving_trace(seed=args.seed)
    wave_size = max(1, args.serving_concurrency)
    await fetch_worker_snapshot(omni, reset_chains=True, include_audit=False)

    all_results: list[dict] = []
    t_suite_start = time.perf_counter()
    for w_idx in range(0, len(trace), wave_size):
        batch = trace[w_idx : w_idx + wave_size]
        reqs = await asyncio.gather(
            *[
                run_request(
                    omni,
                    entry["item"],
                    f"srv-s{entry['slot']}-{entry['kind']}",
                    want_audio=entry["want_audio"],
                    max_tokens=24,
                    kind=entry["kind"],
                )
                for entry in batch
            ]
        )
        all_results.extend(reqs)
        print(
            f"[e2e_serving|{args.mode}] wave {w_idx // wave_size + 1}/{(len(trace) + wave_size - 1) // wave_size} "
            f"kinds={[r['kind'] for r in reqs]} "
            f"cached={[r['cached_tokens'] for r in reqs]} "
            f"TTFT={[round(r['ttft_ms'] or 0, 1) for r in reqs]}",
            flush=True,
        )

    total_wall_s = time.perf_counter() - t_suite_start
    # Allow trailing async saves/evictions to settle before auditing Store
    await asyncio.sleep(2.0)
    final_snap = await fetch_worker_snapshot(omni, reset_chains=False, include_audit=True)

    hot_audio = [r for r in all_results if r["kind"] == "hot_audio"]
    hot_text = [r for r in all_results if r["kind"] == "hot_text"]
    cold_noise = [r for r in all_results if r["kind"] == "cold_noise"]

    # SLO attainment: Text-only TTFT <= 350 ms; Audio TTFA <= 550 ms (cold noise TTFT <= 900 ms)
    slo_met = 0
    for r in all_results:
        if r["kind"] == "hot_text":
            slo_met += int((r["ttft_ms"] or 1e9) <= 350.0)
        elif r["kind"] == "hot_audio":
            slo_met += int((r["first_audio_ms"] or 1e9) <= 550.0)
        else:
            limit = 650.0 if r["modality"] == "text" else 950.0
            metric = r["ttft_ms"] if r["modality"] == "text" else r["first_audio_ms"]
            slo_met += int((metric or 1e9) <= limit)

    summary = {
        "total_requests": len(all_results),
        "total_wall_s": total_wall_s,
        "throughput_req_per_s": len(all_results) / total_wall_s if total_wall_s > 0 else 0.0,
        "slo_goodput_ratio": slo_met / len(all_results) if all_results else 0.0,
        "total_cached_tokens": sum(r["cached_tokens"] for r in all_results),
        "overall_ttft_ms": stats_dict([r["ttft_ms"] for r in all_results if r["ttft_ms"] is not None]),
        "overall_ttfa_ms": stats_dict([r["first_audio_ms"] for r in all_results if r["first_audio_ms"] is not None]),
        "overall_e2e_ms": stats_dict([r["e2e_ms"] for r in all_results]),
        "hot_audio_ttft_ms": stats_dict([r["ttft_ms"] for r in hot_audio if r["ttft_ms"] is not None]),
        "hot_audio_ttfa_ms": stats_dict([r["first_audio_ms"] for r in hot_audio if r["first_audio_ms"] is not None]),
        "hot_text_ttft_ms": stats_dict([r["ttft_ms"] for r in hot_text if r["ttft_ms"] is not None]),
        "cold_noise_ttft_ms": stats_dict([r["ttft_ms"] for r in cold_noise if r["ttft_ms"] is not None]),
        "store_audit": final_snap.get("store_audit", {}),
        "worker_diag": final_snap.get("diag", {}),
    }
    report["e2e_summary"] = summary
    # Strip raw token_ids list in large serving output if needed, keep clean per-request records
    for r in all_results:
        r.pop("token_ids", None)
    report["e2e_requests"] = all_results


async def run_eval(args) -> None:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    pynvml.nvmlInit()
    git_head = os.environ.get("BENCHMARK_GIT_HEAD")
    if not git_head:
        try:
            git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git_head = "unknown"

    namespace = f"ch3-{args.suite}-{args.mode}-{uuid4().hex[:8]}"
    max_conc = max([args.serving_concurrency, *args.concurrency])
    overrides = deployment(
        mode=args.mode,
        store_config=args.store_config,
        namespace=namespace,
        max_concurrency=max_conc,
        kv_bytes=args.kv_mib * 1024**2,
        max_model_len=args.max_model_len,
        max_store_bytes=args.max_store_mib * 1024**2,
        host_hs_bytes=args.host_hs_mib * 1024**2,
        force_min_write_hits=1 if args.suite == "exactness" else None,
    )

    report = {
        "head": git_head,
        "suite": args.suite,
        "mode": args.mode,
        "kv_mib": args.kv_mib,
        "max_store_mib": args.max_store_mib,
        "host_hs_mib": args.host_hs_mib,
        "max_model_len": args.max_model_len,
        "namespace": namespace,
        "packages": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    omni = AsyncOmni(
        model=args.model,
        revision=REVISION,
        deploy_config="qwen3_omni_moe_mrv2.yaml",
        stage_overrides=overrides,
        trust_remote_code=True,
        stage_init_timeout=1800,
        init_timeout=1800,
        log_stats=False,
    )
    try:
        # Warmup both Text-only and Audio kernels with distinct non-colliding warmup keys
        for idx in range(args.warmup):
            await run_request(
                omni,
                build_scaled_prompt(f"warmup-audio-{idx}-{uuid4().hex[:4]}", 512),
                f"warmup-audio-{idx}",
                want_audio=True,
                max_tokens=12,
                kind="warmup",
            )
            await run_request(
                omni,
                build_scaled_prompt(f"warmup-text-{idx}-{uuid4().hex[:4]}", 512),
                f"warmup-text-{idx}",
                want_audio=False,
                max_tokens=12,
                kind="warmup",
            )
        await asyncio.sleep(1.0)
        await fetch_worker_snapshot(omni, reset_chains=True, include_audit=False)

        if args.suite == "exactness":
            await run_exactness_suite(omni, report)
        elif args.suite == "read_matrix":
            await run_read_matrix_suite(omni, args, report)
        elif args.suite == "e2e_serving":
            await run_e2e_serving_suite(omni, args, report)

        report["completed"] = True
    finally:
        omni.shutdown()
        output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("exactness", "read_matrix", "e2e_serving"), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--model", default="/mnt/pfs/wangzhen/workspace/model/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--store-config", default=os.environ.get("MOONCAKE_CONFIG_PATH"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--serving-concurrency", type=int, default=4)
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    parser.add_argument("--max-model-len", type=int, default=8704)
    parser.add_argument("--kv-mib", type=int, default=1280)
    parser.add_argument("--max-store-mib", type=int, default=384)
    parser.add_argument("--host-hs-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
