# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wave 1A Diagnostic Benchmark for CoState (Qwen3-Omni #10107 Baseline).

Covers three strictly controlled M0/M1 calibration & root-cause isolation tasks:
1. ``jit_media``: Isolates first-run Triton/encoder JIT compilation from
   Mooncake write/lookup queueing on multimodal (image/audio) cold & warm turns
   by running explicit multimodal warmup turns with distinct cache keys before
   5 timed repeats.
2. ``long_e0``: Calibrates cold compute, local hit, and Mooncake joint restore
   across 512 / 2048 / 4096 / 8192 token prompt tiers (c=1, c=2, c=4) and
   sweeps ``max_hs_bytes`` (256 MiB default vs. 1024 MiB / 4096 MiB tuned).
3. In-memory connector funnel & write/read micro-stage timing counters
   collected strictly between timed waves (no per-op CUDA synchronization).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path
from urllib.request import urlopen
from uuid import uuid4

import numpy as np
import psutil
import pynvml
import torch
from PIL import Image

from tests.engine import kv_offload_helpers as helpers
from vllm_omni.worker.gpu_ar_worker import GPUARWorker

REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"


class DiagnosticGPUARWorker(GPUARWorker):
    """Collects non-intrusive CPU/Store funnel timings in memory between waves."""

    def init_device(self):
        from safetensors.torch import load as st_load
        from vllm_omni.distributed.kv_transfer.mooncake import connector as conn_mod
        from vllm_omni.distributed.kv_transfer.mooncake import thinker_captures as tc_mod
        from vllm_omni.distributed.kv_transfer.mooncake.connector import (
            JointStoreWorker,
            OmniMooncakeStoreConnector,
        )

        self._diag_lock = threading.Lock()
        self._reset_diag_buffers()

        # 1. Wrap JointStoreWorker.lookup
        orig_lookup = JointStoreWorker.lookup

        def timed_lookup(worker_self, num_tokens, block_hashes):
            t0 = time.perf_counter()
            res = orig_lookup(worker_self, num_tokens, block_hashes)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._diag_lock:
                self._diag["lookup_calls"] += 1
                self._diag["lookup_ms"].append(dt_ms)
                self._diag["lookup_hit_tokens"].append(res.hit_length)
            return res

        JointStoreWorker.lookup = timed_lookup

        # 2. Wrap OmniMooncakeStoreConnector.start_load_kv to track admission funnel
        orig_start_load = OmniMooncakeStoreConnector.start_load_kv

        def timed_start_load(connector_self, forward_context, **kwargs):
            t0 = time.perf_counter()
            orig_start_load(connector_self, forward_context, **kwargs)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with connector_self._lock:
                mem_used = connector_self._memory_used()
                active_reads = len(connector_self._reads)
                active_saves = len(connector_self._saves)
                failed_tokens = [
                    tok for tok, p in connector_self._reads.items() if p.failed and p.reserved_bytes == 0
                ]
            with self._diag_lock:
                self._diag["start_load_ms"].append(dt_ms)
                self._diag["peak_memory_used_bytes"] = max(self._diag["peak_memory_used_bytes"], mem_used)
                self._diag["peak_active_reads"] = max(self._diag["peak_active_reads"], active_reads)
                self._diag["peak_active_saves"] = max(self._diag["peak_active_saves"], active_saves)
                for tok in failed_tokens:
                    if tok not in self._seen_rejected_tokens:
                        self._seen_rejected_tokens.add(tok)
                        self._diag["rejected_reads"] += 1

        OmniMooncakeStoreConnector.start_load_kv = timed_start_load

        # 3. Wrap OmniMooncakeStoreConnector._read_hs
        orig_read_hs = OmniMooncakeStoreConnector._read_hs

        def timed_read_hs(connector_self, attempt, tokens):
            t0 = time.perf_counter()
            out = orig_read_hs(connector_self, attempt, tokens)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._diag_lock:
                self._diag["admitted_reads"] += 1
                self._diag["read_hs_ms"].append(dt_ms)
                self._diag["read_hs_tokens"].append(tokens)
            return out

        OmniMooncakeStoreConnector._read_hs = timed_read_hs

        # 4. Wrap _write_hs_block & _save_hs to isolate H_W2a (encode+put), H_W2b (get+decode_hs), H_W2c (KV poll)
        orig_decode_hs = conn_mod.decode_hs

        def timed_decode_hs(payload, expected):
            t0 = time.perf_counter()
            res = orig_decode_hs(payload, expected)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._diag_lock:
                self._diag["decode_hs_calls"] += 1
                self._diag["decode_hs_ms"].append(dt_ms)
            return res

        conn_mod.decode_hs = timed_decode_hs

        orig_write_hs_block = OmniMooncakeStoreConnector._write_hs_block

        def timed_write_hs_block(connector_self, store, attempt, block):
            t0 = time.perf_counter()
            res = orig_write_hs_block(connector_self, store, attempt, block)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._diag_lock:
                self._diag["write_block_calls"] += 1
                self._diag["write_block_total_ms"].append(dt_ms)
            return res

        OmniMooncakeStoreConnector._write_hs_block = timed_write_hs_block

        orig_save_hs = OmniMooncakeStoreConnector._save_hs

        def timed_save_hs(connector_self, save):
            t0 = time.perf_counter()
            before_blocks = len(save.blocks)
            ok = orig_save_hs(connector_self, save)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._diag_lock:
                self._diag["save_batches"] += 1
                self._diag["save_blocks_total"] += before_blocks
                self._diag["save_hs_total_ms"].append(dt_ms)
                if not ok:
                    self._diag["save_failures"] += 1
            return ok

        OmniMooncakeStoreConnector._save_hs = timed_save_hs

        # 5. Wrap capture_step to isolate H_W1 (Worker main-thread capture & staging)
        orig_capture_step = OmniMooncakeStoreConnector.capture_step

        def timed_capture_step(connector_self, req_id, attempt_id, start_token, prompt_len, tensors):
            t0 = time.perf_counter()
            merged = orig_capture_step(connector_self, req_id, attempt_id, start_token, prompt_len, tensors)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            with self._diag_lock:
                self._diag["capture_step_calls"] += 1
                self._diag["capture_step_ms"].append(dt_ms)
                if merged is not None:
                    inserted = merged["embedding"].shape[0] - tensors["embedding"].shape[0]
                    if inserted > 0:
                        self._diag["restored_rows"].append((req_id, inserted))
            return merged

        OmniMooncakeStoreConnector.capture_step = timed_capture_step
        _ = (st_load, tc_mod)
        super().init_device()

    def _reset_diag_buffers(self):
        self._seen_rejected_tokens = set()
        self._diag = {
            "lookup_calls": 0,
            "lookup_ms": [],
            "lookup_hit_tokens": [],
            "start_load_ms": [],
            "admitted_reads": 0,
            "rejected_reads": 0,
            "read_hs_ms": [],
            "read_hs_tokens": [],
            "capture_step_calls": 0,
            "capture_step_ms": [],
            "write_block_calls": 0,
            "write_block_total_ms": [],
            "decode_hs_calls": 0,
            "decode_hs_ms": [],
            "save_batches": 0,
            "save_blocks_total": 0,
            "save_hs_total_ms": [],
            "save_failures": 0,
            "peak_memory_used_bytes": 0,
            "peak_active_reads": 0,
            "peak_active_saves": 0,
            "restored_rows": [],
        }

    def costate_diag_snapshot(self):
        process = psutil.Process()
        memory = process.memory_full_info()
        with self._diag_lock:
            snap = dict(self._diag)
            self._reset_diag_buffers()
        return {
            "pid": process.pid,
            "rss_bytes": memory.rss,
            "uss_bytes": memory.uss,
            "kv_capacity_tokens": self.cache_config.num_gpu_blocks * self.cache_config.block_size,
            "diag": snap,
        }


def build_scaled_prompt(document: str, target_tokens: int) -> dict:
    """Build deterministic factual prompt scaled to ~target_tokens."""
    base_facts = helpers._FACTS
    # ~18-20 tokens per fact sentence; scale deterministically to hit target_tokens
    needed_facts = max(24, int(target_tokens / 17.2))
    pieces = []
    for idx in range(needed_facts):
        fact = base_facts[idx % len(base_facts)]
        pieces.append(f"Section {idx // len(base_facts)} Fact {idx}: {fact}")
    body = " ".join(pieces)
    text = (
        f"<|im_start|>user\nDocument {document}. {body}\n"
        f"State Fact 0 above in one short sentence.<|im_end|>\n<|im_start|>assistant\n"
    )
    return {"prompt": text}


def build_media_prompt(document: str, modality: str, changed: bool = False, seed_offset: int = 0) -> dict:
    result = build_scaled_prompt(document, 512)
    if modality == "image":
        tag = "<|vision_start|><|image_pad|><|vision_end|>"
        color = (200 - seed_offset * 10, 30, 30) if not changed else (30, 30, 200 - seed_offset * 10)
        media = Image.new("RGB", (112, 112), color)
    else:
        tag = "<|audio_start|><|audio_pad|><|audio_end|>"
        samples = np.arange(16000, dtype=np.float32) / 16000
        freq = (440 + seed_offset * 15) if not changed else (660 + seed_offset * 15)
        media = (0.1 * np.sin(2 * np.pi * freq * samples), 16000)
    result["prompt"] = result["prompt"].replace("user\n", f"user\n{tag}", 1)
    result["multi_modal_data"] = {modality: media}
    return result


def deployment(
    mode: str,
    store_config: str | None,
    namespace: str,
    max_concurrency: int,
    kv_bytes: int,
    max_model_len: int,
    max_hs_bytes: int,
    max_pending_transfers: int,
):
    thinker = {
        "use_v2_model_runner": True,
        "worker_cls": "benchmarks.qwen3_omni.costate_wave1a_diag.DiagnosticGPUARWorker",
        "tensor_parallel_size": 1,
        "max_num_seqs": max_concurrency,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": min(2048, max(256, max_model_len // 2)),
        "enable_chunked_prefill": True,
        "enable_prefix_caching": mode == "local",
        "block_size": 16,
        "gpu_memory_utilization": 0.88,
        "kv_cache_memory_bytes": kv_bytes,
        "revision": REVISION,
    }
    if mode == "mooncake":
        assert store_config is not None
        thinker["omni_kv_config"] = {
            "kv_store_config": {
                "mooncake_config": {
                    "config_file": str(Path(store_config).resolve()),
                    "namespace": namespace,
                    "model_revision": REVISION,
                    "max_hs_bytes": max_hs_bytes,
                    "max_pending_transfers": max_pending_transfers,
                }
            }
        }
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


async def run_single_request(omni, item: dict, request_id: str, max_tokens: int = 32):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    greedy = {**helpers.GREEDY, "max_tokens": max_tokens}
    params = [SamplingParams(**greedy, output_kind=RequestOutputKind.DELTA) for _ in range(omni.num_stages)]
    start = time.perf_counter()
    res = {"request_id": request_id, "ttft_ms": None, "first_audio_ms": None, "cached_tokens": 0}
    tokens, audio, text_parts = [], [], []
    async for out in omni.generate(
        item, request_id=request_id, sampling_params_list=params, output_modalities=["text", "audio"]
    ):
        if out.error:
            raise RuntimeError(out.error)
        elapsed = (time.perf_counter() - start) * 1000.0
        if out.stage_id == 0:
            res["cached_tokens"] = max(res["cached_tokens"], out.num_cached_tokens or 0)
            res["prompt_tokens"] = len(out.prompt_token_ids or []) or res.get("prompt_tokens", 0)
            for completion in out.outputs:
                if completion.token_ids:
                    if res["ttft_ms"] is None:
                        res["ttft_ms"] = elapsed
                    tokens.extend(completion.token_ids)
                    text_parts.append(completion.text)
        if out.final_output_type == "audio":
            value = out.multimodal_output.get("audio")
            if isinstance(value, torch.Tensor) and value.numel():
                if res["first_audio_ms"] is None:
                    res["first_audio_ms"] = elapsed
                audio.append(value)
    res["e2e_ms"] = (time.perf_counter() - start) * 1000.0
    res["text"] = "".join(text_parts)
    res["num_gen_tokens"] = len(tokens)
    waveform = torch.cat([piece.detach().cpu().float().reshape(-1) for piece in audio]) if audio else torch.empty(0)
    res["audio_samples"] = waveform.numel()
    res["audio_sha256"] = hashlib.sha256(waveform.numpy().tobytes()).hexdigest() if waveform.numel() else ""
    res["success"] = bool(tokens and waveform.numel())
    return res


def stats_dict(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None, "sum": 0.0}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "sum": sum(values),
    }


def memory_snapshot(store_metrics_url: str | None) -> dict:
    gpus = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpus.append({"index": index, "used_bytes": mem.used, "total_bytes": mem.total})
    result = {"gpus": gpus}
    if store_metrics_url:
        try:
            with urlopen(store_metrics_url, timeout=5) as response:
                metrics = response.read().decode()
            result["store_metrics"] = {
                name: float(val)
                for line in metrics.splitlines()
                if line and not line.startswith("#")
                for name, val in [line.rsplit(" ", 1)]
                if any(w in name.lower() for w in ("memory", "bytes", "size", "capacity", "object"))
            }
        except Exception as exc:
            result["store_metrics_error"] = str(exc)
    return result


async def run_benchmark(args):
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    pynvml.nvmlInit()
    git_head = os.environ.get("BENCHMARK_GIT_HEAD")
    if not git_head:
        try:
            git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git_head = "unknown"

    namespace = f"w1a-{args.suite}-{args.mode}-{args.max_hs_mib}m-{uuid4().hex[:8]}"
    overrides = deployment(
        mode=args.mode,
        store_config=args.store_config,
        namespace=namespace,
        max_concurrency=max(args.concurrency),
        kv_bytes=args.kv_mib * 1024**2,
        max_model_len=args.max_model_len,
        max_hs_bytes=args.max_hs_mib * 1024**2,
        max_pending_transfers=args.max_pending_transfers,
    )

    report = {
        "head": git_head,
        "suite": args.suite,
        "mode": args.mode,
        "max_hs_mib": args.max_hs_mib,
        "kv_mib": args.kv_mib,
        "max_model_len": args.max_model_len,
        "namespace": namespace,
        "packages": {name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")},
        "waves": [],
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
        # Text warmup
        for idx in range(args.warmup):
            await run_single_request(omni, build_scaled_prompt(f"warmup-text-{idx}", 512), f"warmup-text-{idx}")

        # Multimodal JIT warmup when testing jit_media
        if args.suite == "jit_media":
            for idx in range(2):
                await run_single_request(
                    omni,
                    build_media_prompt(f"warmup-img-{idx}", "image", seed_offset=idx + 10),
                    f"warmup-img-{idx}",
                )
                await run_single_request(
                    omni,
                    build_media_prompt(f"warmup-aud-{idx}", "audio", seed_offset=idx + 10),
                    f"warmup-aud-{idx}",
                )
            # Let background saves settle before resetting diagnostic counters
            await asyncio.sleep(1.5)

        await omni.collective_rpc("costate_diag_snapshot", stage_ids=[0])

        async def run_wave(label: str, repeat: int, concurrency: int, items: list[dict]):
            start = time.perf_counter()
            reqs = await asyncio.gather(
                *[
                    run_single_request(omni, item, f"{label}-r{repeat}-c{concurrency}-i{i}")
                    for i, item in enumerate(items)
                ]
            )
            wall_s = time.perf_counter() - start
            # Give best-effort background save threads a brief moment to record stats for this wave
            await asyncio.sleep(0.2)
            snaps = await omni.collective_rpc("costate_diag_snapshot", stage_ids=[0])
            diag = snaps[0]["diag"] if snaps else {}
            diag_summary = {
                k: stats_dict(v) if isinstance(v, list) and (not v or isinstance(v[0], (int, float))) else v
                for k, v in diag.items()
            }
            wave_row = {
                "case": label,
                "repeat": repeat,
                "concurrency": concurrency,
                "wall_seconds": wall_s,
                "requests": reqs,
                "ttft_stats": stats_dict([r["ttft_ms"] for r in reqs if r["ttft_ms"] is not None]),
                "ttfa_stats": stats_dict([r["first_audio_ms"] for r in reqs if r["first_audio_ms"] is not None]),
                "e2e_stats": stats_dict([r["e2e_ms"] for r in reqs]),
                "thinker_diag": diag_summary,
                "memory": memory_snapshot(args.store_metrics_url),
            }
            report["waves"].append(wave_row)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                f"[{args.mode}|hs={args.max_hs_mib}M] {label} r={repeat} c={concurrency} "
                f"prompt={[r.get('prompt_tokens') for r in reqs]} "
                f"cached={[r['cached_tokens'] for r in reqs]} "
                f"TTFT_mean={wave_row['ttft_stats']['mean']:.1f}ms "
                f"TTFA_mean={wave_row['ttfa_stats']['mean']:.1f}ms "
                f"rej={diag.get('rejected_reads', 0)}",
                flush=True,
            )

        if args.suite == "jit_media":
            for repeat in range(args.repeats):
                for modality in ("image", "audio"):
                    doc = f"postwarmup-{modality}-r{repeat}"
                    await run_wave(
                        f"{modality}_cold_warmed",
                        repeat,
                        1,
                        [build_media_prompt(doc, modality, seed_offset=repeat)],
                    )
                    await run_wave(
                        f"{modality}_full_warmed",
                        repeat,
                        1,
                        [build_media_prompt(doc, modality, seed_offset=repeat)],
                    )
        elif args.suite == "long_e0":
            for target_len in args.prompt_lengths:
                for concurrency in args.concurrency:
                    for repeat in range(args.repeats):
                        docs = [f"len{target_len}-c{concurrency}-r{repeat}-d{i}" for i in range(concurrency)]
                        items = [build_scaled_prompt(d, target_len) for d in docs]
                        await run_wave(f"len{target_len}_cold", repeat, concurrency, items)
                        if args.post_cold_settle_s > 0:
                            await asyncio.sleep(args.post_cold_settle_s)
                        await run_wave(f"len{target_len}_full", repeat, concurrency, items)

        report["completed"] = True
    finally:
        omni.shutdown()
        output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("jit_media", "long_e0"), required=True)
    parser.add_argument("--mode", choices=("off", "local", "mooncake"), required=True)
    parser.add_argument("--model", default="/mnt/pfs/wangzhen/workspace/model/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--store-config", default=os.environ.get("MOONCAKE_CONFIG_PATH"))
    parser.add_argument("--store-metrics-url")
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[512, 2048, 4096, 8192])
    parser.add_argument("--max-model-len", type=int, default=8704)
    parser.add_argument("--kv-mib", type=int, default=2560)
    parser.add_argument("--max-hs-mib", type=int, default=256)
    parser.add_argument("--max-pending-transfers", type=int, default=16)
    parser.add_argument("--post-cold-settle-s", type=float, default=1.5)
    args = parser.parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
