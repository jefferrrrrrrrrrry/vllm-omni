# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""StagePace: Reuse-State and Remaining-Work-Aware Cross-Stage Flow Control & Dynamic Batching.

CoState (Chapter 3) accelerates Stage 0 (Thinker) prefix reuse by 5-13x, turning
cached multi-turn Audio arrivals into short, high-slack bursts while cold long-prompt
arrivals retain tight TTFA headroom. Without cross-stage pacing, those bursts overwhelm
Stage 1 (Talker, which runs with ``recompute_preemption="fail"`` across 11 Omni/TTS
models), break Talker K-step multi-frame decode, and create a static-window dilemma in
Stage 2 (Code2Wav).

StagePace coordinates all three stages through a single zero-RPC control plane:

1. **Mechanism 1 (`TalkerCapacityGuard`, Orchestrator)**:
   Dual-watermark Talker KV headroom + in-flight lease backpressure with path-aware
   Text-only fast-lane bypass (`final_stage_id == 0`) and earliest-deadline-first (EDF)
   Audio pacing queue. Eliminates Talker preemption failures without head-of-line
   blocking Text-only turns on GPU 0.
2. **Mechanism 2 (`SlackBatchController` — Stage 1 Talker)**:
   Tracks per-request chunk-0 readiness timestamps and end-to-end TTFA slack
   (`t_arr + TTFA_SLO - now - est_downstream`). Cold/tight-SLO arrivals bypass
   immediately (`0 ms` defer), while high-slack cached arrivals coalesce into joint
   prefill steps without disarming active K-step multi-frame decode.
3. **Mechanism 3 (`SlackBatchController` — Stage 2 Code2Wav)**:
   Dual-slack dynamic coalescing window bounded jointly by runnable first-chunk TTFA
   slack (`S_first`) and active-stream playback continuity slack (`S_cont`), plus
   selective Express lane that only isolates urgent first chunks so high-slack cached
   first chunks pack into wide CUDA-graph vocoder batches.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

STAGEPACE_INFO_KEY = "stagepace"


@dataclass(frozen=True)
class StagePaceConfig:
    """Configuration for StagePace cross-stage pacing and dynamic batching.

    Disabled by default (`enabled=False`) so upstream behavior is strictly preserved
    unless opted in via ``additional_config["stagepace"]`` or connector ``extra``.
    """

    enabled: bool = False

    # Mechanism 1: Orchestrator cross-stage backpressure + Text-only fast lane
    enable_backpressure: bool = True
    enable_text_fastlane: bool = True
    enable_edf_queue: bool = True
    talker_max_inflight_reqs: int = 8
    talker_kv_high_watermark: float = 0.78
    talker_kv_low_watermark: float = 0.65
    talker_kv_token_capacity: int = 0
    talker_est_decode_tokens: int = 160
    default_ttfa_slo_ms: float = 650.0

    # Mechanism 2: Stage 0/1 Reuse-State & Remaining-Work Ordering + Talker K-step Coalescing
    enable_work_aware_reorder: bool = True
    enable_talker_slack_coalesce: bool = True
    talker_min_batch: int = 2
    talker_min_running: int = 1
    talker_max_wait_ms: float = 22.0
    talker_min_defer_slack_ms: float = 140.0
    talker_est_first_audio_ms: float = 95.0
    talker_slack_fraction: float = 0.12

    # Mechanism 3: Stage 2 Code2Wav dual-slack dynamic coalescing + selective express
    enable_code2wav_dual_slack: bool = True
    code2wav_target_batch: int = 4
    code2wav_max_wait_ms: float = 18.0
    code2wav_express_slack_threshold_ms: float = 120.0
    code2wav_continuation_guard_ms: float = 150.0
    code2wav_est_vocoder_ms: float = 25.0
    code2wav_slack_fraction: float = 0.10

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> StagePaceConfig:
        if not raw or not isinstance(raw, Mapping):
            return cls()
        # Accept either top-level dict or nested under "stagepace"
        cfg = raw.get("stagepace") if isinstance(raw.get("stagepace"), Mapping) else raw
        if not isinstance(cfg, Mapping) or not cfg.get("enabled", False):
            return cls()
        defaults = cls()
        high_wm = float(cfg.get("talker_kv_high_watermark", defaults.talker_kv_high_watermark))
        low_wm = float(cfg.get("talker_kv_low_watermark", defaults.talker_kv_low_watermark))
        if not (0.0 < low_wm <= high_wm <= 1.0):
            raise ValueError(
                f"StagePace watermarks must satisfy 0 < low ({low_wm}) <= high ({high_wm}) <= 1.0"
            )
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            enable_backpressure=bool(cfg.get("enable_backpressure", defaults.enable_backpressure)),
            enable_text_fastlane=bool(cfg.get("enable_text_fastlane", defaults.enable_text_fastlane)),
            enable_edf_queue=bool(cfg.get("enable_edf_queue", defaults.enable_edf_queue)),
            talker_max_inflight_reqs=max(
                1, int(cfg.get("talker_max_inflight_reqs", defaults.talker_max_inflight_reqs))
            ),
            talker_kv_high_watermark=high_wm,
            talker_kv_low_watermark=low_wm,
            talker_kv_token_capacity=max(
                0, int(cfg.get("talker_kv_token_capacity", defaults.talker_kv_token_capacity))
            ),
            talker_est_decode_tokens=max(
                0, int(cfg.get("talker_est_decode_tokens", defaults.talker_est_decode_tokens))
            ),
            default_ttfa_slo_ms=max(1.0, float(cfg.get("default_ttfa_slo_ms", defaults.default_ttfa_slo_ms))),
            enable_work_aware_reorder=bool(
                cfg.get("enable_work_aware_reorder", defaults.enable_work_aware_reorder)
            ),
            enable_talker_slack_coalesce=bool(
                cfg.get("enable_talker_slack_coalesce", defaults.enable_talker_slack_coalesce)
            ),
            talker_min_batch=max(1, int(cfg.get("talker_min_batch", defaults.talker_min_batch))),
            talker_min_running=max(0, int(cfg.get("talker_min_running", defaults.talker_min_running))),
            talker_max_wait_ms=max(0.0, float(cfg.get("talker_max_wait_ms", defaults.talker_max_wait_ms))),
            talker_min_defer_slack_ms=max(
                0.0, float(cfg.get("talker_min_defer_slack_ms", defaults.talker_min_defer_slack_ms))
            ),
            talker_est_first_audio_ms=max(
                0.0, float(cfg.get("talker_est_first_audio_ms", defaults.talker_est_first_audio_ms))
            ),
            talker_slack_fraction=max(
                0.0, min(1.0, float(cfg.get("talker_slack_fraction", defaults.talker_slack_fraction)))
            ),
            enable_code2wav_dual_slack=bool(
                cfg.get("enable_code2wav_dual_slack", defaults.enable_code2wav_dual_slack)
            ),
            code2wav_target_batch=max(1, int(cfg.get("code2wav_target_batch", defaults.code2wav_target_batch))),
            code2wav_max_wait_ms=max(0.0, float(cfg.get("code2wav_max_wait_ms", defaults.code2wav_max_wait_ms))),
            code2wav_express_slack_threshold_ms=max(
                0.0,
                float(
                    cfg.get(
                        "code2wav_express_slack_threshold_ms",
                        defaults.code2wav_express_slack_threshold_ms,
                    )
                ),
            ),
            code2wav_continuation_guard_ms=max(
                0.0,
                float(cfg.get("code2wav_continuation_guard_ms", defaults.code2wav_continuation_guard_ms)),
            ),
            code2wav_est_vocoder_ms=max(
                0.0, float(cfg.get("code2wav_est_vocoder_ms", defaults.code2wav_est_vocoder_ms))
            ),
            code2wav_slack_fraction=max(
                0.0, min(1.0, float(cfg.get("code2wav_slack_fraction", defaults.code2wav_slack_fraction)))
            ),
        )

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> StagePaceConfig:
        """Read StagePaceConfig from vllm_config.additional_config or stage_connector_config."""
        if vllm_config is None:
            return cls()
        additional = getattr(vllm_config, "additional_config", None)
        if isinstance(additional, Mapping) and isinstance(additional.get("stagepace"), Mapping):
            return cls.from_mapping(additional)
        model_config = getattr(vllm_config, "model_config", None)
        connector_cfg = getattr(model_config, "stage_connector_config", None) if model_config is not None else None
        extra = (
            connector_cfg.get("extra", connector_cfg)
            if isinstance(connector_cfg, Mapping)
            else getattr(connector_cfg, "extra", None)
        )
        if isinstance(extra, Mapping) and isinstance(extra.get("stagepace"), Mapping):
            return cls.from_mapping(extra)
        return cls()


@dataclass(frozen=True)
class StagePaceReqMeta:
    """Global arrival and SLO metadata propagated from Orchestrator to Stages 1 & 2."""

    t_arr_wall: float
    ttfa_slo_ms: float
    est_prompt_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_arr_wall": float(self.t_arr_wall),
            "ttfa_slo_ms": float(self.ttfa_slo_ms),
            "est_prompt_tokens": int(self.est_prompt_tokens),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, default_slo_ms: float) -> StagePaceReqMeta | None:
        t_arr = raw.get("t_arr_wall")
        if t_arr is None:
            return None
        try:
            t_arr_f = float(t_arr)
            slo_f = float(raw.get("ttfa_slo_ms", default_slo_ms))
            plen = int(raw.get("est_prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(t_arr_f) and math.isfinite(slo_f)):
            return None
        return cls(t_arr_wall=t_arr_f, ttfa_slo_ms=slo_f, est_prompt_tokens=max(0, plen))


def extract_and_pop_stagepace_meta(
    request: Any, *, default_slo_ms: float = 650.0
) -> StagePaceReqMeta | None:
    """Pop ``stagepace`` metadata off ``request.additional_information`` if present.

    Stripping the scheduler-only key on arrival keeps the model runner's
    additional_information payload completely unchanged while preserving the
    metadata in scheduler memory across async-chunk payload replacements.
    """
    info = getattr(request, "additional_information", None)
    if info is None:
        return None

    raw_entry: Any = None
    if isinstance(info, dict):
        raw_entry = info.pop(STAGEPACE_INFO_KEY, None)
        if not info:
            request.additional_information = None
    elif hasattr(info, "entries") and isinstance(info.entries, dict):
        entry = info.entries.pop(STAGEPACE_INFO_KEY, None)
        if entry is not None:
            raw_entry = getattr(entry, "scalar_data", entry)
        if not info.entries:
            request.additional_information = None
    else:
        return None

    if isinstance(raw_entry, Mapping):
        return StagePaceReqMeta.from_mapping(raw_entry, default_slo_ms=default_slo_ms)
    return None


@dataclass
class TalkerCapacityGuard:
    """Mechanism 1: Dual-watermark Stage 1 KV guard + Text-only fast-lane bypass."""

    config: StagePaceConfig
    observed_kv_usage: float = 0.0
    observed_running_reqs: int = 0
    observed_waiting_reqs: int = 0
    _throttling: bool = False
    _leases: dict[str, int] = field(default_factory=dict)

    # White-box diagnostic counters
    audio_admitted: int = 0
    audio_throttled: int = 0
    text_fastlane_bypassed: int = 0
    text_throttled: int = 0
    max_inflight_leases: int = 0
    max_observed_kv_usage: float = 0.0
    total_throttle_wait_ms: float = 0.0
    throttled_completions: int = 0

    @property
    def active_lease_count(self) -> int:
        return len(self._leases)

    @property
    def leased_kv_tokens(self) -> int:
        return sum(self._leases.values())

    @property
    def effective_kv_usage(self) -> float:
        est = 0.0
        cap = self.config.talker_kv_token_capacity
        if cap > 0:
            est = self.leased_kv_tokens / float(cap)
        return max(self.observed_kv_usage, est)

    def observe_talker_stats(
        self,
        *,
        kv_cache_usage: float | None = None,
        num_running: int | None = None,
        num_waiting: int | None = None,
    ) -> None:
        if kv_cache_usage is not None and math.isfinite(kv_cache_usage):
            usage = max(0.0, min(1.0, float(kv_cache_usage)))
            self.observed_kv_usage = usage
            if usage > self.max_observed_kv_usage:
                self.max_observed_kv_usage = usage
            if usage >= self.config.talker_kv_high_watermark:
                self._throttling = True
            elif usage <= self.config.talker_kv_low_watermark:
                self._throttling = False
        if num_running is not None:
            self.observed_running_reqs = max(0, int(num_running))
        if num_waiting is not None:
            self.observed_waiting_reqs = max(0, int(num_waiting))

    def should_admit_request(
        self,
        req_id: str,
        *,
        is_text_only: bool,
        est_kv_tokens: int = 0,
    ) -> bool:
        if not self.config.enabled or not self.config.enable_backpressure:
            return True

        # Path-aware fast-lane: Text-only requests terminate at Stage 0 and consume
        # zero Stage 1 Talker KV blocks -> bypass Stage 1 backpressure completely!
        if is_text_only and self.config.enable_text_fastlane:
            self.text_fastlane_bypassed += 1
            return True

        # Always guarantee forward progress when no leased request is in flight.
        if not self._leases:
            return True

        if len(self._leases) >= self.config.talker_max_inflight_reqs:
            self._throttling = True
            return False

        watermark = (
            self.config.talker_kv_low_watermark
            if self._throttling
            else self.config.talker_kv_high_watermark
        )
        if self.observed_kv_usage >= watermark:
            self._throttling = True
            return False

        cap = self.config.talker_kv_token_capacity
        if cap > 0:
            projected = (self.leased_kv_tokens + max(0, est_kv_tokens)) / float(cap)
            if projected > watermark:
                self._throttling = True
                return False

        self._throttling = False
        return True

    def acquire_lease(self, req_id: str, est_kv_tokens: int, *, is_text_only: bool = False) -> None:
        if not self.config.enabled or not self.config.enable_backpressure:
            return
        if is_text_only and self.config.enable_text_fastlane:
            return
        self._leases[req_id] = max(0, int(est_kv_tokens))
        if len(self._leases) > self.max_inflight_leases:
            self.max_inflight_leases = len(self._leases)
        if not is_text_only:
            self.audio_admitted += 1

    def record_throttled(self, *, is_text_only: bool = False) -> None:
        if is_text_only:
            self.text_throttled += 1
        else:
            self.audio_throttled += 1

    def record_throttle_wait(self, wait_ms: float) -> None:
        if wait_ms > 0.0 and math.isfinite(wait_ms):
            self.total_throttle_wait_ms += wait_ms
            self.throttled_completions += 1

    def release_lease(self, req_id: str) -> bool:
        removed = self._leases.pop(req_id, None) is not None
        if not self._leases:
            # Deadlock-proof reset when all leased requests have drained from Stage 1.
            self.observed_kv_usage = 0.0
            self._throttling = False
        return removed

    def snapshot_diagnostics(self) -> dict[str, Any]:
        return {
            "active_leases": len(self._leases),
            "leased_kv_tokens": self.leased_kv_tokens,
            "observed_kv_usage": round(self.observed_kv_usage, 4),
            "max_observed_kv_usage": round(self.max_observed_kv_usage, 4),
            "max_inflight_leases": self.max_inflight_leases,
            "audio_admitted": self.audio_admitted,
            "audio_throttled": self.audio_throttled,
            "text_fastlane_bypassed": self.text_fastlane_bypassed,
            "text_throttled": self.text_throttled,
            "mean_throttle_wait_ms": (
                round(self.total_throttle_wait_ms / self.throttled_completions, 2)
                if self.throttled_completions > 0
                else 0.0
            ),
        }


class SlackBatchController:
    """Mechanisms 2 & 3: Unified TTFA-Slack + Playback-Slack scheduler controller."""

    def __init__(self, config: StagePaceConfig) -> None:
        self.config = config
        # Stage 0 / Stage 1 work-aware queue ordering diagnostics
        self.waiting_reorders = 0
        # Stage 1 Talker diagnostics
        self.talker_deferrals = 0
        self.talker_urgent_bypasses = 0
        self.talker_target_batch_admissions = 0
        self.talker_max_wait_admissions = 0
        self.talker_kstep_steps = 0
        self.talker_single_frame_steps = 0
        # Stage 2 Code2Wav diagnostics
        self.code2wav_zero_wait_urgent = 0
        self.code2wav_zero_wait_cont_guard = 0
        self.code2wav_coalesced_waits = 0
        self.code2wav_express_urgent = 0
        self.code2wav_express_suppressed_hot = 0
        self.code2wav_steps = 0
        self.code2wav_total_batch_rows = 0

    def remaining_ttfa_slack_ms(
        self,
        meta: StagePaceReqMeta | None,
        *,
        fallback_arrival_wall: float,
        now_wall: float,
        est_downstream_ms: float,
    ) -> float:
        t_arr = meta.t_arr_wall if meta is not None else fallback_arrival_wall
        slo_ms = meta.ttfa_slo_ms if meta is not None else self.config.default_ttfa_slo_ms
        elapsed_ms = max(0.0, (now_wall - t_arr) * 1000.0)
        return slo_ms - elapsed_ms - est_downstream_ms

    def order_waiting_requests(
        self,
        waiting_reqs: Sequence[Any],
        *,
        stage_id: int,
        seen_hashes: set[bytes],
        block_size: int = 16,
        is_stage_zero_final_fn: Any = None,
        ready_ts_map: dict[str, float] | None = None,
        now_wall: float | None = None,
    ) -> list[Any]:
        """Mechanism 2A: Reuse-state and remaining-work-aware waiting queue ordering (SRPT + Aging).

        - On Stage 0 (Thinker): estimates each waiting request's uncached remaining prefill
          tokens using ``seen_hashes`` (prefix blocks already computed on Stage 0) and applies
          a fast-lane priority weight to Text-only requests (``final_stage_id == 0``). This
          prevents 20-token CoState prefix-cache hits and fast Text-only turns from being
          head-of-line blocked behind 5000-token cold misses while sharing the same 4096-token
          chunked-prefill batch.
        - On Stage 1 (Talker): orders ready chunk-0 requests by remaining Talker prompt tokens
          with anti-starvation aging so short hot utterances are not trapped behind a giant
          cold Talker prefill.
        """
        cfg = self.config
        reqs = list(waiting_reqs)
        if len(reqs) <= 1 or not cfg.enabled or not cfg.enable_work_aware_reorder:
            return reqs

        t_wall = time.time() if now_wall is None else now_wall
        bsize = max(1, int(block_size))
        scored: list[tuple[float, int, Any]] = []

        for idx, req in enumerate(reqs):
            rid = getattr(req, "request_id", "")
            ptoks = getattr(req, "prompt_token_ids", None)
            prompt_len = len(ptoks) if ptoks is not None else int(getattr(req, "num_prompt_tokens", 0) or 0)
            computed = int(getattr(req, "num_computed_tokens", 0) or 0)

            if stage_id == 0:
                est_cached = computed
                if seen_hashes:
                    bhashes = getattr(req, "block_hashes", None) or ()
                    matched_blks = 0
                    for bh in bhashes:
                        if bytes(bh) not in seen_hashes:
                            break
                        matched_blks += 1
                    est_cached = max(est_cached, matched_blks * bsize)
                rem_tokens = max(bsize, prompt_len - est_cached)
                is_text = bool(is_stage_zero_final_fn(req)) if callable(is_stage_zero_final_fn) else False
                mod_weight = 0.30 if (is_text and cfg.enable_text_fastlane) else 1.0
                arr_ts = float(getattr(req, "arrival_time", t_wall) or t_wall)
                wait_s = max(0.0, t_wall - arr_ts)
                # Anti-starvation aging: 3200 tokens/s aging credit guarantees bounded wait for cold prompts
                score = float(rem_tokens) * mod_weight - 3200.0 * wait_s
            else:
                ready_ts = (
                    ready_ts_map.setdefault(rid, t_wall)
                    if ready_ts_map is not None
                    else float(getattr(req, "arrival_time", t_wall) or t_wall)
                )
                rem_tokens = max(1, prompt_len - computed)
                wait_s = max(0.0, t_wall - ready_ts)
                score = float(rem_tokens) - 3200.0 * wait_s

            scored.append((score, idx, req))

        scored.sort(key=lambda item: (item[0], item[1]))
        ordered = [item[2] for item in scored]
        if any(a is not b for a, b in zip(ordered, reqs)):
            self.waiting_reorders += 1
        return ordered

    def should_defer_talker_prefill(
        self,
        waiting_reqs: Sequence[Any],
        *,
        running_count: int,
        ready_ts_map: dict[str, float],
        meta_map: Mapping[str, StagePaceReqMeta],
        now_wall: float | None = None,
    ) -> tuple[bool, str]:
        """Mechanism 2: Decide whether Stage 1 Talker should briefly hold waiting prefills.

        While Talker is executing K-step multi-frame decode (`running_count >= min_running`),
        deferring high-slack cached arrivals for a bounded window coalesces scattered single
        arrivals into one joint prefill step and preserves K-step decode continuity. Any
        cold/tight-SLO request immediately breaks the hold (`0 ms` defer).
        """
        cfg = self.config
        if not cfg.enabled or not cfg.enable_talker_slack_coalesce:
            return False, "disabled"
        if cfg.talker_min_batch <= 1 or cfg.talker_max_wait_ms <= 0.0:
            return False, "disabled"
        if running_count < cfg.talker_min_running or not waiting_reqs:
            return False, "idle"
        if len(waiting_reqs) >= cfg.talker_min_batch:
            self.talker_target_batch_admissions += 1
            return False, "target_batch_reached"

        now = time.time() if now_wall is None else now_wall
        min_slack_ms = float("inf")
        oldest_ready = now
        for req in waiting_reqs:
            rid = req.request_id
            ready_ts = ready_ts_map.setdefault(rid, now)
            if ready_ts < oldest_ready:
                oldest_ready = ready_ts
            slack_ms = self.remaining_ttfa_slack_ms(
                meta_map.get(rid),
                fallback_arrival_wall=float(getattr(req, "arrival_time", ready_ts)),
                now_wall=now,
                est_downstream_ms=cfg.talker_est_first_audio_ms,
            )
            if slack_ms < min_slack_ms:
                min_slack_ms = slack_ms

        if min_slack_ms < cfg.talker_min_defer_slack_ms:
            self.talker_urgent_bypasses += 1
            return False, "urgent_bypass"

        allowed_wait_ms = min(cfg.talker_max_wait_ms, min_slack_ms * cfg.talker_slack_fraction)
        waited_ms = max(0.0, (now - oldest_ready) * 1000.0)
        if waited_ms < allowed_wait_ms:
            self.talker_deferrals += 1
            return True, "coalesce"

        self.talker_max_wait_admissions += 1
        return False, "max_wait_reached"

    def compute_code2wav_wait_s(
        self,
        runnable_ids: Sequence[str],
        *,
        requests_by_id: Mapping[str, Any],
        chunk_started: set[str],
        meta_map: Mapping[str, StagePaceReqMeta],
        stream_audio: Mapping[str, list[float]],
        now_wall: float | None = None,
        now_mono: float | None = None,
    ) -> tuple[float, str]:
        """Mechanism 3A: Dual-slack dynamic coalescing window `W_dyn` (seconds) for Stage 2.

        Bounded simultaneously by:
        - `S_first`: remaining TTFA slack of runnable first chunks (`0` if any is urgent).
        - `S_cont`: remaining unplayed audio buffer of runnable continuation chunks
          minus `code2wav_continuation_guard_ms` (`0` if any stream risks underrun).
        """
        cfg = self.config
        if not cfg.enabled or not cfg.enable_code2wav_dual_slack or cfg.code2wav_max_wait_ms <= 0.0:
            return 0.0, "disabled"
        if not runnable_ids:
            return cfg.code2wav_max_wait_ms / 1000.0, "idle"

        t_wall = time.time() if now_wall is None else now_wall
        t_mono = time.monotonic() if now_mono is None else now_mono
        bound_ms = cfg.code2wav_max_wait_ms
        guard_s = cfg.code2wav_continuation_guard_ms / 1000.0

        for rid in runnable_ids:
            if rid not in chunk_started:
                req = requests_by_id.get(rid)
                fallback_arr = float(getattr(req, "arrival_time", t_wall)) if req is not None else t_wall
                slack_ms = self.remaining_ttfa_slack_ms(
                    meta_map.get(rid),
                    fallback_arrival_wall=fallback_arr,
                    now_wall=t_wall,
                    est_downstream_ms=cfg.code2wav_est_vocoder_ms,
                )
                if slack_ms <= cfg.code2wav_express_slack_threshold_ms:
                    self.code2wav_zero_wait_urgent += 1
                    return 0.0, "urgent_first_chunk"
                bound_ms = min(bound_ms, max(0.0, slack_ms * cfg.code2wav_slack_fraction))
            else:
                audio = stream_audio.get(rid)
                if audio is None:
                    self.code2wav_zero_wait_cont_guard += 1
                    return 0.0, "continuation_no_credit"
                cont_headroom_s = (audio[0] + audio[1] - t_mono) - guard_s
                if cont_headroom_s <= 0.0:
                    self.code2wav_zero_wait_cont_guard += 1
                    return 0.0, "continuation_guard"
                bound_ms = min(bound_ms, cont_headroom_s * 1000.0)

        if bound_ms <= 0.0:
            return 0.0, "slack_exhausted"
        self.code2wav_coalesced_waits += 1
        return bound_ms / 1000.0, "dual_slack_coalesce"

    def should_code2wav_express(
        self,
        waiting_reqs: Sequence[Any],
        *,
        chunk_started: set[str],
        meta_map: Mapping[str, StagePaceReqMeta],
        now_wall: float | None = None,
    ) -> tuple[bool, str]:
        """Mechanism 3B: Selective first-chunk Express lane.

        Only urgent/cold first chunks (`S_first <= express_slack_threshold_ms`) trigger
        isolated Express steps; high-slack cached first chunks stay in unified wide batches.
        """
        cfg = self.config
        if not cfg.enabled or not cfg.enable_code2wav_dual_slack:
            return True, "disabled"

        t_wall = time.time() if now_wall is None else now_wall
        has_first_chunk = False
        for req in waiting_reqs:
            rid = req.request_id
            if rid in chunk_started:
                continue
            if getattr(req, "num_in_flight_tokens", 0) > 0:
                continue
            if len(getattr(req, "prompt_token_ids", ())) <= getattr(req, "num_computed_tokens", 0):
                continue
            has_first_chunk = True
            slack_ms = self.remaining_ttfa_slack_ms(
                meta_map.get(rid),
                fallback_arrival_wall=float(getattr(req, "arrival_time", t_wall)),
                now_wall=t_wall,
                est_downstream_ms=cfg.code2wav_est_vocoder_ms,
            )
            if slack_ms <= cfg.code2wav_express_slack_threshold_ms:
                self.code2wav_express_urgent += 1
                return True, "urgent_first_chunk"

        if has_first_chunk:
            self.code2wav_express_suppressed_hot += 1
            return False, "hot_batched"
        return False, "no_first_chunk"

    def snapshot_diagnostics(self) -> dict[str, Any]:
        avg_batch = (
            round(self.code2wav_total_batch_rows / self.code2wav_steps, 3)
            if self.code2wav_steps > 0
            else 0.0
        )
        return {
            "waiting_reorders": self.waiting_reorders,
            "talker_deferrals": self.talker_deferrals,
            "talker_urgent_bypasses": self.talker_urgent_bypasses,
            "talker_target_batch_admissions": self.talker_target_batch_admissions,
            "talker_max_wait_admissions": self.talker_max_wait_admissions,
            "talker_kstep_steps": self.talker_kstep_steps,
            "talker_single_frame_steps": self.talker_single_frame_steps,
            "code2wav_zero_wait_urgent": self.code2wav_zero_wait_urgent,
            "code2wav_zero_wait_cont_guard": self.code2wav_zero_wait_cont_guard,
            "code2wav_coalesced_waits": self.code2wav_coalesced_waits,
            "code2wav_express_urgent": self.code2wav_express_urgent,
            "code2wav_express_suppressed_hot": self.code2wav_express_suppressed_hot,
            "code2wav_steps": self.code2wav_steps,
            "code2wav_avg_batch_size": avg_batch,
        }
