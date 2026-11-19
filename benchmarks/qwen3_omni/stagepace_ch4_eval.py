# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chapter 4 (StagePace) & Chapter 5 (2x2 CoState x StagePace) Evaluation Suite.

Evaluates cross-stage flow control & dual-slack dynamic batching on
Qwen3-Omni-30B-A3B-Instruct across:
  - Cache modes (--cache-mode): ``local_only``, ``pr10107``, ``costate``
  - Pacing modes (--pace-mode):
      * ``f0_baseline``: Upstream blind FIFO (StagePace disabled)
      * ``f1_static``:   Global concurrency cap without Text-only fast-lane or M2/M3
      * ``stagepace_m1``: Mechanism 1 (Talker KV Guard + Text-only Fast-Lane + EDF)
      * ``stagepace_full``: Full StagePace (M1 + M2 Talker K-step Coalescing + M3 Code2Wav Dual-Slack)

Measures:
  - Talker preemption failure count / rate (``recompute_preemption="fail"``)
  - Mixed-modality latency: Text-only TTFT (HOL-blocking isolation), Hot Audio TTFA, Cold Audio TTFA
  - Real-time audio continuity: inter-chunk gap P95 & playback underrun rate
  - System throughput (req/s) and end-to-end SLO Goodput (failed requests strictly fail SLO)
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pynvml

from benchmarks.qwen3_omni.costate_ch3_eval import (
    REVISION,
    build_scaled_prompt,
    stats_dict,
)
from tests.engine import kv_offload_helpers as helpers

CACHE_MODES = ("local_only", "pr10107", "costate")
PACE_MODES = ("f0_baseline", "f1_static", "stagepace_m1", "stagepace_full")


def build_stagepace_dict(args) -> dict:
    mode = args.pace_mode
    token_cap = args.talker_kv_token_cap
    if token_cap <= 0 and args.talker_kv_mib > 0:
        # Exact physical capacity of Qwen3-Omni Talker KV pool: 20.0 KiB/token (13,088 tokens per 256 MiB)
        token_cap = max(2048, int((args.talker_kv_mib * 1024) / 20))
    if mode == "f0_baseline":
        return {"enabled": False}
    if mode == "f1_static":
        return {
            "enabled": True,
            "enable_backpressure": True,
            "enable_text_fastlane": False,
            "enable_edf_queue": False,
            "talker_max_inflight_reqs": args.talker_max_inflight,
            "talker_kv_high_watermark": args.talker_kv_high_wm,
            "talker_kv_low_watermark": args.talker_kv_low_wm,
            "talker_kv_token_capacity": token_cap,
            "default_ttfa_slo_ms": args.ttfa_slo_ms,
            "enable_talker_slack_coalesce": False,
            "enable_code2wav_dual_slack": False,
        }
    if mode == "stagepace_m1":
        return {
            "enabled": True,
            "enable_backpressure": True,
            "enable_text_fastlane": True,
            "enable_edf_queue": True,
            "talker_max_inflight_reqs": args.talker_max_inflight,
            "talker_kv_high_watermark": args.talker_kv_high_wm,
            "talker_kv_low_watermark": args.talker_kv_low_wm,
            "talker_kv_token_capacity": token_cap,
            "default_ttfa_slo_ms": args.ttfa_slo_ms,
            "enable_talker_slack_coalesce": False,
            "enable_code2wav_dual_slack": False,
        }
    return {
        "enabled": True,
        "enable_backpressure": True,
        "enable_text_fastlane": True,
        "enable_edf_queue": True,
        "talker_max_inflight_reqs": args.talker_max_inflight,
        "talker_kv_high_watermark": args.talker_kv_high_wm,
        "talker_kv_low_watermark": args.talker_kv_low_wm,
        "talker_kv_token_capacity": token_cap,
        "default_ttfa_slo_ms": args.ttfa_slo_ms,
        "enable_talker_slack_coalesce": True,
        "talker_min_batch": 2,
        "talker_min_running": 1,
        "talker_max_wait_ms": 22.0,
        "talker_min_defer_slack_ms": 140.0,
        "enable_code2wav_dual_slack": True,
        "code2wav_target_batch": 4,
        "code2wav_max_wait_ms": 18.0,
        "code2wav_express_slack_threshold_ms": 120.0,
        "code2wav_continuation_guard_ms": 150.0,
    }


def build_deployment(args, namespace: str) -> dict:
    sp_cfg = build_stagepace_dict(args)
    enable_local = args.cache_mode in ("local_only", "costate")
    max_seqs = max(args.concurrency, 12)

    thinker: dict = {
        "use_v2_model_runner": True,
        "tensor_parallel_size": 1,
        "max_num_seqs": max_seqs,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": min(4096, max(512, args.max_model_len // 2)),
        "enable_chunked_prefill": True,
        "enable_prefix_caching": enable_local,
        "block_size": 16,
        "gpu_memory_utilization": 0.88,
        "kv_cache_memory_bytes": args.thinker_kv_mib * 1024**2,
        "revision": REVISION,
        "additional_config": {"stagepace": sp_cfg},
    }

    if args.cache_mode in ("pr10107", "costate"):
        assert args.store_config, "--store-config required for pr10107 / costate modes"
        mc_cfg: dict[str, object] = {
            "config_file": str(Path(args.store_config).resolve()),
            "namespace": namespace,
            "model_revision": REVISION,
            "max_pending_transfers": 16,
        }
        if args.cache_mode == "pr10107":
            mc_cfg["max_hs_bytes"] = 256 * 1024 * 1024
            mc_cfg["enable_costate"] = False
        else:
            mc_cfg.update(
                max_hs_bytes=1024 * 1024 * 1024,
                enable_costate=True,
                enable_local_tier=True,
                enable_modality_decouple=True,
                enable_cost_gate=True,
                min_write_hits=2,
                eviction_policy="costate_utility",
                sync_write_truncation=True,
                max_store_bytes=args.max_store_mib * 1024**2,
                host_hs_cache_bytes=args.host_hs_mib * 1024**2,
            )
        thinker["omni_kv_config"] = {"kv_store_config": {"mooncake_config": mc_cfg}}

    talker: dict = {
        "max_num_seqs": max_seqs,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": 0.35,
        "revision": REVISION,
        "subtalker_sampling_params": {"do_sample": False},
        "additional_config": {"stagepace": sp_cfg},
    }
    if args.talker_kv_mib > 0:
        talker["kv_cache_memory_bytes"] = args.talker_kv_mib * 1024**2

    code2wav: dict = {
        "max_num_seqs": max_seqs,
        "gpu_memory_utilization": 0.20,
        "revision": REVISION,
        "additional_config": {"stagepace": sp_cfg},
    }

    return helpers.stage_overrides(
        lmcache=False,
        thinker_extra=thinker,
        downstream_extra={"1": talker, "2": code2wav},
    )


def build_stagepace_trace(*, num_requests: int, seed: int) -> list[dict]:
    """Mixed high-concurrency Omni burst trace:
    - 55% Hot Audio turns (1.6K-2.8K shared multi-turn prefixes -> high CoState hit slack)
    - 25% Fast Text-only turns (1.6K shared prefixes -> tests fast-lane vs HOL blocking)
    - 20% Cold Long Audio turns (4K-6K one-off prompts -> tight TTFA slack, tests urgent bypass)
    """
    rng = random.Random(seed)
    hot_sessions = [
        ("sp-hot-0", 1600),
        ("sp-hot-1", 2048),
        ("sp-hot-2", 2800),
        ("sp-hot-3", 1600),
    ]
    trace: list[dict] = []
    for idx in range(num_requests):
        slot = idx % 20
        if slot in (2, 7, 13, 18):
            cold_len = 4096 if slot in (2, 13) else 6144
            trace.append(
                {
                    "slot": idx,
                    "kind": "cold_audio",
                    "want_audio": True,
                    "max_tokens": 36,
                    "item": build_scaled_prompt(
                        f"sp-cold-{idx}",
                        cold_len,
                        turn_suffix=f"Cold query {idx}: explain briefly in two sentences.",
                    ),
                }
            )
        elif slot in (4, 8, 11, 15, 19):
            sess_id, plen = hot_sessions[idx % len(hot_sessions)]
            trace.append(
                {
                    "slot": idx,
                    "kind": "hot_text",
                    "want_audio": False,
                    "max_tokens": 24,
                    "item": build_scaled_prompt(
                        sess_id,
                        plen,
                        turn_suffix=f"Text turn {idx}: list item {rng.randint(1, 50)} in one line.",
                    ),
                }
            )
        else:
            sess_id, plen = hot_sessions[idx % len(hot_sessions)]
            trace.append(
                {
                    "slot": idx,
                    "kind": "hot_audio",
                    "want_audio": True,
                    "max_tokens": 40,
                    "item": build_scaled_prompt(
                        sess_id,
                        plen,
                        turn_suffix=f"Audio turn {idx}: answer question {rng.randint(1, 50)} concisely.",
                    ),
                }
            )
    return trace


async def run_one_request(omni, entry: dict, *, ttfa_slo_ms: float) -> dict:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    req_id = f"sp-s{entry['slot']}-{entry['kind']}"
    want_audio = entry["want_audio"]
    max_tokens = entry["max_tokens"]
    greedy = {**helpers.GREEDY, "max_tokens": max_tokens}
    params = [SamplingParams(**greedy, output_kind=RequestOutputKind.DELTA) for _ in range(omni.num_stages)]
    modalities = ["text", "audio"] if want_audio else ["text"]

    prompt = dict(entry["item"])
    start = time.perf_counter()
    res: dict = {
        "request_id": req_id,
        "kind": entry["kind"],
        "modality": "audio" if want_audio else "text",
        "ok": True,
        "error": None,
        "ttft_ms": None,
        "first_audio_ms": None,
        "e2e_ms": None,
        "cached_tokens": 0,
        "prompt_tokens": 0,
        "audio_chunks": 0,
        "max_audio_gap_ms": 0.0,
        "underrun_events": 0,
    }
    audio_chunk_ts: list[float] = []
    emitted_audio_s = 0.0
    first_audio_mono: float | None = None

    try:
        async for out in omni.generate(
            prompt,
            req_id,
            sampling_params_list=params,
            output_modalities=modalities,
        ):
            now = time.perf_counter()
            if out.final_output_type == "text" and out.request_output is not None:
                ro = out.request_output
                if ro.prompt_token_ids:
                    res["prompt_tokens"] = len(ro.prompt_token_ids)
                if getattr(ro, "num_cached_tokens", None):
                    res["cached_tokens"] = int(ro.num_cached_tokens)
                if res["ttft_ms"] is None and ro.outputs and ro.outputs[0].token_ids:
                    res["ttft_ms"] = (now - start) * 1000.0
            elif out.final_output_type == "audio" and out.request_output is not None:
                mm = getattr(out.request_output, "multimodal_output", None) or {}
                wav = mm.get("audio") if "audio" in mm else mm.get("model_outputs")
                if wav is not None and hasattr(wav, "numel") and wav.numel() > 0:
                    if first_audio_mono is None:
                        first_audio_mono = now
                        res["first_audio_ms"] = (now - start) * 1000.0
                    else:
                        gap_ms = (now - audio_chunk_ts[-1]) * 1000.0
                        if gap_ms > res["max_audio_gap_ms"]:
                            res["max_audio_gap_ms"] = gap_ms
                        # Underrun if wall time since first audio exceeded previously emitted playback duration
                        if (now - first_audio_mono) > emitted_audio_s + 0.01:
                            res["underrun_events"] += 1
                    audio_chunk_ts.append(now)
                    res["audio_chunks"] += 1
                    sr = mm.get("sr", 24000)
                    sr_val = int(sr.reshape(-1)[0]) if hasattr(sr, "reshape") else int(sr or 24000)
                    emitted_audio_s += float(wav.numel()) / max(1, sr_val)
        res["e2e_ms"] = (time.perf_counter() - start) * 1000.0
        if want_audio and res["first_audio_ms"] is None:
            res["ok"] = False
            res["error"] = "no_audio_emitted"
    except Exception as exc:
        res["ok"] = False
        res["error"] = f"{type(exc).__name__}: {exc}"
        res["e2e_ms"] = (time.perf_counter() - start) * 1000.0
    return res


async def run_eval(args) -> None:
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    pynvml.nvmlInit()
    git_head = os.environ.get("BENCHMARK_GIT_HEAD")
    if not git_head:
        try:
            git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            git_head = "unknown"

    namespace = f"ch4-{args.cache_mode}-{args.pace_mode}-{uuid4().hex[:8]}"
    overrides = build_deployment(args, namespace)
    report: dict = {
        "head": git_head,
        "cache_mode": args.cache_mode,
        "pace_mode": args.pace_mode,
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "thinker_kv_mib": args.thinker_kv_mib,
        "talker_kv_mib": args.talker_kv_mib,
        "ttfa_slo_ms": args.ttfa_slo_ms,
        "text_slo_ms": args.text_slo_ms,
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
        # Warmup hot sessions once so caches and CUDA graphs are warm before burst serving
        for idx in range(2):
            await run_one_request(
                omni,
                {
                    "slot": -(idx + 1),
                    "kind": "warmup_audio",
                    "want_audio": True,
                    "max_tokens": 16,
                    "item": build_scaled_prompt(f"sp-warm-{idx}-{uuid4().hex[:4]}", 512),
                },
                ttfa_slo_ms=args.ttfa_slo_ms,
            )
        await asyncio.sleep(1.0)

        trace = build_stagepace_trace(num_requests=args.num_requests, seed=args.seed)
        sem = asyncio.Semaphore(max(1, args.concurrency))
        all_results: list[dict] = []

        async def _bounded(entry: dict, delay_s: float) -> dict:
            if delay_s > 0:
                await asyncio.sleep(delay_s)
            async with sem:
                return await run_one_request(omni, entry, ttfa_slo_ms=args.ttfa_slo_ms)

        t0 = time.perf_counter()
        # Submit in realistic burst waves (concurrency arrivals per wave, staggered lightly)
        tasks = [
            asyncio.create_task(_bounded(entry, (i // args.concurrency) * args.wave_interval_s + (i % args.concurrency) * 0.015))
            for i, entry in enumerate(trace)
        ]
        all_results = await asyncio.gather(*tasks)
        wall_s = time.perf_counter() - t0

        ok_reqs = [r for r in all_results if r["ok"]]
        failed_reqs = [r for r in all_results if not r["ok"]]
        hot_audio = [r for r in ok_reqs if r["kind"] == "hot_audio"]
        cold_audio = [r for r in ok_reqs if r["kind"] == "cold_audio"]
        hot_text = [r for r in ok_reqs if r["kind"] == "hot_text"]

        slo_met = 0
        for r in all_results:
            if not r["ok"]:
                continue
            if r["kind"] == "hot_text":
                slo_met += int((r["ttft_ms"] or 1e9) <= args.text_slo_ms)
            elif r["kind"] == "hot_audio":
                slo_met += int((r["first_audio_ms"] or 1e9) <= args.ttfa_slo_ms)
            else:
                slo_met += int((r["first_audio_ms"] or 1e9) <= args.cold_ttfa_slo_ms)

        orch_diag = {}
        orch = getattr(getattr(omni, "engine", None), "orchestrator", None)
        if orch is not None and hasattr(orch, "get_stagepace_diagnostics"):
            orch_diag = orch.get_stagepace_diagnostics()

        report["summary"] = {
            "total_requests": len(all_results),
            "completed_requests": len(ok_reqs),
            "failed_requests": len(failed_reqs),
            "failure_rate": len(failed_reqs) / len(all_results) if all_results else 0.0,
            "wall_s": wall_s,
            "throughput_req_per_s": len(ok_reqs) / wall_s if wall_s > 0 else 0.0,
            "slo_goodput_ratio": slo_met / len(all_results) if all_results else 0.0,
            "hot_text_ttft_ms": stats_dict([r["ttft_ms"] for r in hot_text if r["ttft_ms"] is not None]),
            "hot_audio_ttft_ms": stats_dict([r["ttft_ms"] for r in hot_audio if r["ttft_ms"] is not None]),
            "hot_audio_ttfa_ms": stats_dict([r["first_audio_ms"] for r in hot_audio if r["first_audio_ms"] is not None]),
            "cold_audio_ttfa_ms": stats_dict([r["first_audio_ms"] for r in cold_audio if r["first_audio_ms"] is not None]),
            "audio_max_gap_ms": stats_dict([r["max_audio_gap_ms"] for r in (hot_audio + cold_audio)]),
            "total_underrun_events": sum(r["underrun_events"] for r in all_results),
            "orchestrator_stagepace_diag": orch_diag,
            "sample_errors": [r["error"] for r in failed_reqs[:5]],
        }
        report["requests"] = all_results
        report["completed"] = True
        print(
            f"[stagepace|{args.cache_mode}|{args.pace_mode}] ok={len(ok_reqs)}/{len(all_results)} "
            f"fail={len(failed_reqs)} tput={report['summary']['throughput_req_per_s']:.2f} req/s "
            f"goodput={report['summary']['slo_goodput_ratio']*100:.1f}% "
            f"text_ttft={report['summary']['hot_text_ttft_ms']['mean'] or 0:.1f}ms "
            f"hot_ttfa={report['summary']['hot_audio_ttfa_ms']['mean'] or 0:.1f}ms "
            f"cold_ttfa={report['summary']['cold_audio_ttfa_ms']['mean'] or 0:.1f}ms",
            flush=True,
        )
    finally:
        omni.shutdown()
        output.write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-mode", choices=CACHE_MODES, default="costate")
    parser.add_argument("--pace-mode", choices=PACE_MODES, required=True)
    parser.add_argument("--model", default="/mnt/pfs/wangzhen/workspace/model/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--store-config", default=os.environ.get("MOONCAKE_CONFIG_PATH"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--num-requests", type=int, default=48)
    parser.add_argument("--wave-interval-s", type=float, default=0.35)
    parser.add_argument("--max-model-len", type=int, default=8704)
    parser.add_argument("--thinker-kv-mib", type=int, default=1536)
    parser.add_argument("--talker-kv-mib", type=int, default=256)
    parser.add_argument("--max-store-mib", type=int, default=512)
    parser.add_argument("--host-hs-mib", type=int, default=256)
    parser.add_argument("--talker-max-inflight", type=int, default=6)
    parser.add_argument("--talker-kv-high-wm", type=float, default=0.78)
    parser.add_argument("--talker-kv-low-wm", type=float, default=0.62)
    parser.add_argument("--talker-kv-token-cap", type=int, default=0)
    parser.add_argument("--ttfa-slo-ms", type=float, default=750.0)
    parser.add_argument("--cold-ttfa-slo-ms", type=float, default=1200.0)
    parser.add_argument("--text-slo-ms", type=float, default=350.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
