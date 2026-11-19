# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for StagePace cross-stage flow control & dual-slack dynamic batching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.stagepace import (
    STAGEPACE_INFO_KEY,
    SlackBatchController,
    StagePaceConfig,
    StagePaceReqMeta,
    TalkerCapacityGuard,
    extract_and_pop_stagepace_meta,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_stagepace_config_defaults_and_validation() -> None:
    default_cfg = StagePaceConfig.from_mapping(None)
    assert not default_cfg.enabled

    vcfg = SimpleNamespace(
        additional_config={
            "stagepace": {
                "enabled": True,
                "talker_max_inflight_reqs": 6,
                "talker_kv_high_watermark": 0.80,
                "talker_kv_low_watermark": 0.60,
                "default_ttfa_slo_ms": 550.0,
            }
        },
        model_config=SimpleNamespace(stage_connector_config=None),
    )
    cfg = StagePaceConfig.from_vllm_config(vcfg)
    assert cfg.enabled
    assert cfg.talker_max_inflight_reqs == 6
    assert cfg.talker_kv_high_watermark == pytest.approx(0.80)
    assert cfg.talker_kv_low_watermark == pytest.approx(0.60)
    assert cfg.default_ttfa_slo_ms == pytest.approx(550.0)

    with pytest.raises(ValueError, match="watermarks"):
        StagePaceConfig.from_mapping(
            {"enabled": True, "talker_kv_low_watermark": 0.85, "talker_kv_high_watermark": 0.75}
        )


def test_extract_and_pop_stagepace_meta_strips_payload_cleanly() -> None:
    req_dict = SimpleNamespace(
        additional_information={
            STAGEPACE_INFO_KEY: {"t_arr_wall": 1000.5, "ttfa_slo_ms": 500.0, "est_prompt_tokens": 2048},
            "other_key": 123,
        }
    )
    meta = extract_and_pop_stagepace_meta(req_dict, default_slo_ms=650.0)
    assert meta == StagePaceReqMeta(t_arr_wall=1000.5, ttfa_slo_ms=500.0, est_prompt_tokens=2048)
    assert STAGEPACE_INFO_KEY not in req_dict.additional_information
    assert req_dict.additional_information == {"other_key": 123}

    # Payload carrying only stagepace collapses to None cleanly
    payload_obj = SimpleNamespace(
        entries={
            STAGEPACE_INFO_KEY: SimpleNamespace(scalar_data={"t_arr_wall": 2000.0, "ttfa_slo_ms": 600.0})
        }
    )
    req_payload = SimpleNamespace(additional_information=payload_obj)
    meta2 = extract_and_pop_stagepace_meta(req_payload)
    assert meta2 == StagePaceReqMeta(t_arr_wall=2000.0, ttfa_slo_ms=600.0, est_prompt_tokens=0)
    assert req_payload.additional_information is None


def test_talker_capacity_guard_dual_watermark_and_text_fastlane() -> None:
    cfg = StagePaceConfig(
        enabled=True,
        enable_backpressure=True,
        enable_text_fastlane=True,
        talker_max_inflight_reqs=3,
        talker_kv_high_watermark=0.75,
        talker_kv_low_watermark=0.60,
        talker_kv_token_capacity=10_000,
    )
    guard = TalkerCapacityGuard(cfg)

    # First two audio requests fit within inflight cap and KV token lease capacity
    assert guard.should_admit_request("a1", is_text_only=False, est_kv_tokens=3000)
    guard.acquire_lease("a1", 3000)
    assert guard.should_admit_request("a2", is_text_only=False, est_kv_tokens=3000)
    guard.acquire_lease("a2", 3000)

    # Third audio request would project 9000/10000 = 0.90 > 0.75 high watermark -> throttled!
    assert not guard.should_admit_request("a3", is_text_only=False, est_kv_tokens=3000)

    # Text-only request bypasses Stage 1 Talker backpressure immediately (fast-lane)!
    assert guard.should_admit_request("t1", is_text_only=True, est_kv_tokens=0)
    assert guard.text_fastlane_bypassed == 1

    # Real-time Talker KV watermark hysteresis: 0.70 is still > low_watermark (0.60) while throttling
    guard.observe_talker_stats(kv_cache_usage=0.80, num_running=2, num_waiting=0)
    assert not guard.should_admit_request("a4", is_text_only=False, est_kv_tokens=500)
    guard.observe_talker_stats(kv_cache_usage=0.68, num_running=2, num_waiting=0)
    assert not guard.should_admit_request("a4", is_text_only=False, est_kv_tokens=500)

    # Dropping below low watermark AND releasing a1 resumes admission
    guard.release_lease("a1")
    guard.observe_talker_stats(kv_cache_usage=0.55, num_running=1, num_waiting=0)
    assert guard.should_admit_request("a3", is_text_only=False, est_kv_tokens=2000)

    # Deadlock-proof reset when all active leases finish
    guard.observe_talker_stats(kv_cache_usage=0.95, num_running=1, num_waiting=0)
    guard.release_lease("a2")
    assert guard.active_lease_count == 0
    assert guard.observed_kv_usage == 0.0
    assert guard.should_admit_request("a5", is_text_only=False, est_kv_tokens=3000)


def test_talker_capacity_guard_static_ablation_throttles_text() -> None:
    cfg = StagePaceConfig(
        enabled=True,
        enable_backpressure=True,
        enable_text_fastlane=False,
        talker_max_inflight_reqs=1,
    )
    guard = TalkerCapacityGuard(cfg)
    assert guard.should_admit_request("a1", is_text_only=False, est_kv_tokens=1000)
    guard.acquire_lease("a1", 1000)
    # Without fast-lane (F1-Static), Text-only is head-of-line blocked behind Audio
    assert not guard.should_admit_request("t1", is_text_only=True, est_kv_tokens=0)


def test_stage1_talker_slack_coalescing_protects_kstep_and_cold_requests() -> None:
    cfg = StagePaceConfig(
        enabled=True,
        enable_talker_slack_coalesce=True,
        talker_min_batch=3,
        talker_min_running=1,
        talker_max_wait_ms=20.0,
        talker_min_defer_slack_ms=140.0,
        talker_est_first_audio_ms=95.0,
        talker_slack_fraction=0.12,
    )
    ctrl = SlackBatchController(cfg)
    now = 1000.0
    # Cached hot request: finished Stage 0 in 50 ms -> slack = 650 - 50 - 95 = 505 ms (>= 140 ms)
    hot_req = SimpleNamespace(request_id="hot-1", arrival_time=now - 0.05)
    meta_map = {"hot-1": StagePaceReqMeta(t_arr_wall=now - 0.05, ttfa_slo_ms=650.0)}
    ready_ts: dict[str, float] = {}

    # Even though request arrived 50ms ago (stale prewarm arrival_time), chunk-0 just became ready -> defer!
    defer, reason = ctrl.should_defer_talker_prefill(
        [hot_req], running_count=2, ready_ts_map=ready_ts, meta_map=meta_map, now_wall=now
    )
    assert defer and reason == "coalesce"

    # Once chunk-0 has waited >= max_wait_ms (20 ms), admit immediately
    defer2, reason2 = ctrl.should_defer_talker_prefill(
        [hot_req], running_count=2, ready_ts_map=ready_ts, meta_map=meta_map, now_wall=now + 0.025
    )
    assert not defer2 and reason2 == "max_wait_reached"

    # Cold long request: spent 520 ms in Stage 0 -> slack = 650 - 520 - 95 = 35 ms (< 140 ms) -> 0 ms defer!
    cold_req = SimpleNamespace(request_id="cold-1", arrival_time=now - 0.52)
    meta_map["cold-1"] = StagePaceReqMeta(t_arr_wall=now - 0.52, ttfa_slo_ms=650.0)
    ready_ts_cold: dict[str, float] = {}
    defer_cold, reason_cold = ctrl.should_defer_talker_prefill(
        [hot_req, cold_req], running_count=2, ready_ts_map=ready_ts_cold, meta_map=meta_map, now_wall=now
    )
    assert not defer_cold and reason_cold == "urgent_bypass"


def test_stage2_code2wav_dual_slack_wait_and_selective_express() -> None:
    cfg = StagePaceConfig(
        enabled=True,
        enable_code2wav_dual_slack=True,
        code2wav_target_batch=4,
        code2wav_max_wait_ms=18.0,
        code2wav_express_slack_threshold_ms=120.0,
        code2wav_continuation_guard_ms=150.0,
        code2wav_est_vocoder_ms=25.0,
        code2wav_slack_fraction=0.10,
    )
    ctrl = SlackBatchController(cfg)
    now_wall = 2000.0
    now_mono = 500.0

    hot_req = SimpleNamespace(
        request_id="hot-first", arrival_time=now_wall - 0.08, num_in_flight_tokens=0, prompt_token_ids=[1], num_computed_tokens=0
    )
    cold_req = SimpleNamespace(
        request_id="cold-first", arrival_time=now_wall - 0.55, num_in_flight_tokens=0, prompt_token_ids=[1], num_computed_tokens=0
    )
    cont_req = SimpleNamespace(
        request_id="cont-1", arrival_time=now_wall - 1.0, num_in_flight_tokens=0, prompt_token_ids=[1] * 25, num_computed_tokens=0
    )
    reqs = {"hot-first": hot_req, "cold-first": cold_req, "cont-1": cont_req}
    meta = {
        "hot-first": StagePaceReqMeta(t_arr_wall=now_wall - 0.08, ttfa_slo_ms=650.0),   # slack = 545 ms
        "cold-first": StagePaceReqMeta(t_arr_wall=now_wall - 0.55, ttfa_slo_ms=650.0),  # slack = 75 ms <= 120 ms
    }
    chunk_started = {"cont-1"}

    # 1. Hot first chunk + healthy continuation (0.40s unplayed audio > 0.15s guard) -> coalesce up to 18 ms
    stream_audio_safe = {"cont-1": [now_mono - 0.10, 0.50]}  # 0.40 s remaining
    wait_s, reason = ctrl.compute_code2wav_wait_s(
        ["hot-first", "cont-1"],
        requests_by_id=reqs,
        chunk_started=chunk_started,
        meta_map=meta,
        stream_audio=stream_audio_safe,
        now_wall=now_wall,
        now_mono=now_mono,
    )
    assert reason == "dual_slack_coalesce"
    assert wait_s == pytest.approx(0.018)

    # Selective Express suppresses isolated batch-1 step for hot first chunk
    express_hot, exp_reason = ctrl.should_code2wav_express(
        [hot_req], chunk_started=chunk_started, meta_map=meta, now_wall=now_wall
    )
    assert not express_hot and exp_reason == "hot_batched"

    # 2. Urgent cold first chunk -> 0 ms wait AND triggers selective Express!
    wait_cold, reason_cold = ctrl.compute_code2wav_wait_s(
        ["cold-first", "cont-1"],
        requests_by_id=reqs,
        chunk_started=chunk_started,
        meta_map=meta,
        stream_audio=stream_audio_safe,
        now_wall=now_wall,
        now_mono=now_mono,
    )
    assert wait_cold == 0.0 and reason_cold == "urgent_first_chunk"
    express_cold, exp_cold_reason = ctrl.should_code2wav_express(
        [hot_req, cold_req], chunk_started=chunk_started, meta_map=meta, now_wall=now_wall
    )
    assert express_cold and exp_cold_reason == "urgent_first_chunk"

    # 3. Low playback buffer continuation (0.10s < 0.15s guard) -> 0 ms wait even with hot first chunk!
    stream_audio_low = {"cont-1": [now_mono - 0.40, 0.50]}  # 0.10 s remaining < 0.15 s guard
    wait_low, reason_low = ctrl.compute_code2wav_wait_s(
        ["hot-first", "cont-1"],
        requests_by_id=reqs,
        chunk_started=chunk_started,
        meta_map=meta,
        stream_audio=stream_audio_low,
        now_wall=now_wall,
        now_mono=now_mono,
    )
    assert wait_low == 0.0 and reason_low == "continuation_guard"


def test_stage0_and_stage1_work_aware_waiting_queue_reorder() -> None:
    cfg = StagePaceConfig(enabled=True, enable_work_aware_reorder=True, enable_text_fastlane=True)
    ctrl = SlackBatchController(cfg)
    now = 3000.0
    shared_hashes = [f"h-{i}".encode() for i in range(100)]  # 1600 cached tokens
    seen = set(shared_hashes)

    # Cold 5200-token Audio arrived first, followed by hot 1624-token Audio (1600 cached -> 24 rem)
    # and hot 1624-token Text-only (1600 cached -> 24 * 0.30 = 7.2 effective rem)
    r_cold = SimpleNamespace(
        request_id="cold-5k",
        prompt_token_ids=[1] * 5200,
        num_computed_tokens=0,
        block_hashes=[b"cold-0", b"cold-1"],
        arrival_time=now - 0.02,
        is_text=False,
    )
    r_hot_audio = SimpleNamespace(
        request_id="hot-audio",
        prompt_token_ids=[1] * 1624,
        num_computed_tokens=0,
        block_hashes=shared_hashes + [b"tail-a"],
        arrival_time=now - 0.01,
        is_text=False,
    )
    r_hot_text = SimpleNamespace(
        request_id="hot-text",
        prompt_token_ids=[1] * 1624,
        num_computed_tokens=0,
        block_hashes=shared_hashes + [b"tail-t"],
        arrival_time=now,
        is_text=True,
    )

    ordered = ctrl.order_waiting_requests(
        [r_cold, r_hot_audio, r_hot_text],
        stage_id=0,
        seen_hashes=seen,
        block_size=16,
        is_stage_zero_final_fn=lambda r: r.is_text,
        now_wall=now,
    )
    assert [r.request_id for r in ordered] == ["hot-text", "hot-audio", "cold-5k"]
    assert ctrl.waiting_reorders == 1

    # Anti-starvation aging: if cold-5k has waited 2.0 seconds (6400 token credit), it jumps to front!
    r_cold_aged = SimpleNamespace(
        request_id="cold-aged",
        prompt_token_ids=[1] * 5200,
        num_computed_tokens=0,
        block_hashes=[b"cold-0"],
        arrival_time=now - 2.0,
        is_text=False,
    )
    ordered_aged = ctrl.order_waiting_requests(
        [r_hot_audio, r_cold_aged],
        stage_id=0,
        seen_hashes=seen,
        block_size=16,
        is_stage_zero_final_fn=lambda r: r.is_text,
        now_wall=now,
    )
    assert [r.request_id for r in ordered_aged] == ["cold-aged", "hot-audio"]
