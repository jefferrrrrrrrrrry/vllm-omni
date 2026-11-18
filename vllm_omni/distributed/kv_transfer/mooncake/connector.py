# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bulk KV/HS loading for Qwen3-Omni turn serving on MRv2.

The native Mooncake connector owns KV copies and their GPU block references.
This adapter adds a joint lookup on its worker, request-attempt identities,
bounded CPU capture storage, and delayed receive completion. Native connector
integration targets vLLM revision 4ba787ec9959e566d3b587a796dff07673324da5.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from copy import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorTransferResults,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import MooncakeStoreConnector
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    MooncakeLookupResult,
    MooncakeStoreConnectorMetadata,
    MooncakeStoreWorkerMetadata,
    TailKeyBoundary,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import MooncakeStoreWorker
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

from .config import MooncakeConfig
from .costate import (
    COSTATE_HS_SENTINEL_GROUP_ID,
    CoStateReqPlan,
    CoStateWorkerMetadata,
    HostHSCache,
    OnlineCostGate,
    UnifiedPrefixDirectory,
    is_text_only_request,
    split_costate_lookup_result,
)
from .thinker_captures import CaptureTensors, HSBlockSpec, RequestHS, decode_hs, encode_hs, hs_key, prefix_digest

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = init_logger(__name__)
_INIT_LOCK = threading.Lock()
_HS_HEADER_ALLOWANCE = 16392
# One Store round trip fetches the HS objects and completion records of up to
# this many payload bytes, far below the client's default 4 GiB buffer.
_READ_BATCH_BYTES = 16 * 1024 * 1024
_KV_POLL_SECONDS = 0.005
# The load path and library version do not change cached values;
# model_revision identifies the weights.
_UNSTABLE_CONFIG_KEYS = frozenset({"_name_or_path", "transformers_version"})


def _stable_config(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _stable_config(item) for key, item in value.items() if key not in _UNSTABLE_CONFIG_KEYS}
    if isinstance(value, list):
        return [_stable_config(item) for item in value]
    return value


def cache_namespace(
    settings: MooncakeConfig,
    hf_config: PretrainedConfig,
    hidden_layer: int,
    dtype: torch.dtype,
    kv_spec: FullAttentionSpec,
    kv_cache_layout: str,
) -> str:
    """Digest every setting that can change stored KV or HS values."""
    identity = {
        "namespace": settings.namespace,
        "revision": settings.model_revision,
        "stage": "thinker",
        "format": 1,
        "layer": hidden_layer,
        "config": _stable_config(hf_config.to_dict()),
        "dtype": str(dtype),
        "kv_spec": repr(kv_spec),
        "kv_cache_layout": kv_cache_layout,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def _lookup_queue_drained() -> None:
    """Marker job: it runs only after every lookup queued before it has finished."""


class JointStoreWorker(MooncakeStoreWorker):
    """Use the existing lookup RPC to return joint, contiguous availability."""

    enable_costate: bool = False

    def __init__(
        self,
        config: VllmConfig,
        kv_config: KVCacheConfig,
        namespace: str,
        max_tokens: int,
        *,
        enable_costate: bool = False,
    ) -> None:
        self.hs_namespace = namespace
        self.max_hs_tokens = max_tokens
        self.enable_costate = enable_costate
        super().__init__(config, kv_config)
        if self.block_size != self.hash_block_size:
            raise ValueError("Mooncake KV/HS currently requires equal KV and hash block sizes")

    def lookup(self, num_tokens: int, block_hashes: Sequence[BlockHash]) -> MooncakeLookupResult:
        result = super().lookup(num_tokens, block_hashes)
        if self.enable_costate:
            kv_hit = result.hit_length - result.hit_length % self.block_size
            if kv_hit <= 0:
                return MooncakeLookupResult(0)
            end = min(kv_hit, self.max_hs_tokens)
            end -= end % self.block_size
            hashes = block_hashes[: end // self.block_size]
            joint_matched = 0
            if hashes:
                keys = [name for bh in hashes for name in (hs_key(self.hs_namespace, bh), hs_key(self.hs_namespace, bh) + ":ready")]
                try:
                    exists = self.store.batch_is_exist(keys)
                    for index in range(len(hashes)):
                        pair = exists[2 * index : 2 * index + 2]
                        if len(pair) != 2 or not all(value == 1 for value in pair):
                            break
                        joint_matched += self.block_size
                except (RuntimeError, OSError) as exc:
                    logger.warning("Mooncake HS lookup failed: %s", exc)
                    joint_matched = 0
            base_boundaries = tuple(replace(boundary, num_tokens=kv_hit) for boundary in result.tail_key_boundaries)
            sentinel = TailKeyBoundary(group_id=COSTATE_HS_SENTINEL_GROUP_ID, num_tokens=joint_matched)
            return MooncakeLookupResult(kv_hit, (*base_boundaries, sentinel))

        end = min(result.hit_length, self.max_hs_tokens)
        end -= end % self.block_size
        hashes = block_hashes[: end // self.block_size]
        keys: list[str] = []
        for block_hash in hashes:
            key = hs_key(self.hs_namespace, block_hash)
            keys.extend((key, key + ":ready"))
        if not keys:
            return MooncakeLookupResult(0)
        try:
            exists = self.store.batch_is_exist(keys)
        except (RuntimeError, OSError) as exc:
            logger.warning("Mooncake HS lookup failed: %s", exc)
            return MooncakeLookupResult(0)
        matched = 0
        for index in range(len(hashes)):
            pair = exists[2 * index : 2 * index + 2]
            if len(pair) != 2 or not all(value == 1 for value in pair):
                break
            matched += self.block_size
        return MooncakeLookupResult(
            matched,
            tuple(replace(boundary, num_tokens=matched) for boundary in result.tail_key_boundaries) if matched else (),
        )


class CoStateStoreScheduler(MooncakeStoreScheduler):
    """Enforces CoState synchronous KV write gating before block pinning."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        super().__init__(vllm_config, kv_cache_config)
        self.save_kv_limits: dict[str, int] = {}

    def _apply_current_save_block_ids(
        self,
        meta: MooncakeStoreConnectorMetadata,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if self.save_kv_limits:
            kept = []
            for req_meta in meta.requests:
                if req_meta.can_save and req_meta.req_id in self.save_kv_limits:
                    limit = self.save_kv_limits[req_meta.req_id]
                    if limit <= req_meta.token_ids_start:
                        req_meta.can_save = False
                        req_meta.token_len_chunk = 0
                        req_meta.publish_partial_tail = False
                        req_meta.boundary_puts = None
                        if (tracker := self._request_trackers.get(req_meta.req_id)) is not None:
                            tracker.num_saved_tokens = req_meta.token_ids_start
                        if req_meta.load_spec is None:
                            continue
                    elif limit < req_meta.token_len_chunk:
                        req_meta.token_len_chunk = limit
                        req_meta.publish_partial_tail = False
                        if req_meta.token_ids is not None:
                            req_meta.token_ids = req_meta.token_ids[: limit - req_meta.token_ids_start]
                        if (tracker := self._request_trackers.get(req_meta.req_id)) is not None:
                            tracker.num_saved_tokens = limit
                kept.append(req_meta)
            meta.requests = kept
        super()._apply_current_save_block_ids(meta, scheduler_output)


@dataclass
class Attempt:
    """An engine request ID plus a unique retry/preemption generation.

    Ordinary serving already assigns unique internal IDs when a client reuses
    an external ID. Native completion messages must keep that engine identity;
    ``token`` additionally fences successive loads within one engine request.
    """

    req_id: str
    token: str
    prompt_len: int
    hashes: list[bytes] = field(default_factory=list)
    rows: RequestHS | None = None
    cancelled: bool = False
    is_text_only: bool = False
    local_kv_tokens: int = 0
    ext_load_tokens: int = 0
    ext_load_hs: bool = True
    save_store_hs_tokens: int = -1
    save_host_hs: bool = False
    promote_from_host_hs_tokens: int = 0
    promoted_host_hs: bool = False
    host_stage_cursor: int = 0
    host_partial: CaptureTensors | None = None
    host_partial_rows: int = 0


@dataclass
class PendingRead:
    attempt: Attempt
    reserved_bytes: int
    deadline: float
    future: Future[dict[str, torch.Tensor]] | None = None
    kv_done: bool = False
    failed: bool = False
    failure_reported: bool = False
    completion_reported: bool = False
    block_ids: frozenset[int] = frozenset()
    hs_needed: bool = True
    started_at: float = 0.0
    ext_kv_blocks: int = 0
    ext_hs_blocks: int = 0


@dataclass
class PendingSave:
    """Complete prompt blocks queued for saving, with their staging budget."""

    attempt: Attempt
    blocks: list[tuple[int, int, CaptureTensors]]
    reserved_bytes: int


class JointMetadata(MooncakeStoreConnectorMetadata):
    """Native KV transfer metadata plus Omni request-attempt ownership."""

    def __init__(
        self,
        native: MooncakeStoreConnectorMetadata,
        attempts: dict[str, str],
        finished: set[str],
        cancelled: set[str],
        req_plans: dict[str, CoStateReqPlan] | None = None,
        evict_hashes: list[bytes] | None = None,
    ) -> None:
        super().__init__(native.unfinished_request_ids, native.preempted_req_ids)
        self.requests = native.requests
        self.omni_attempts = attempts
        self.omni_finished = finished
        self.omni_cancelled = cancelled
        self.omni_req_plans: dict[str, CoStateReqPlan] = req_plans or {}
        self.omni_evict_hashes: list[bytes] = evict_hashes or []


class OmniMooncakeStoreConnector(MooncakeStoreConnector):
    """Opt-in TP=1 full-attention adapter. All reads precede computation."""

    _costate_dir: UnifiedPrefixDirectory | None = None
    _cost_gate: OnlineCostGate | None = None
    _host_hs_cache: HostHSCache | None = None

    def __init__(
        self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config: KVCacheConfig | None = None
    ) -> None:
        self._closed = False
        self._lock = threading.RLock()
        self._read_executor: ThreadPoolExecutor | None = None
        self._save_executor: ThreadPoolExecutor | None = None
        self._attempts: dict[str, Attempt] = {}
        self._delivered_until: dict[str, int] = {}
        self._reads: dict[str, PendingRead] = {}
        self._saves: dict[Future[bool], PendingSave] = {}
        self._reserved_bytes = 0
        self._scheduler_attempts: dict[str, str] = {}
        self._lookup_started: dict[str, float] = {}
        self._lookup_probe: Future[None] | None = None
        self._bypass: set[str] = set()
        self._retired: set[str] = set()
        self._cancelled: set[str] = set()
        self.connector_worker = None
        self.connector_scheduler = None
        self._costate_dir: UnifiedPrefixDirectory | None = None
        self._cost_gate: OnlineCostGate | None = None
        self._host_hs_cache: HostHSCache | None = None
        self._active_req_plans: dict[str, CoStateReqPlan] = {}
        self._ext_loaded_reqs: set[str] = set()
        self._pending_evict_hashes: list[bytes] = []
        self._pending_worker_meta = CoStateWorkerMetadata()
        self.settings = MooncakeConfig.from_vllm_config(vllm_config)
        self._validate_supported(vllm_config, kv_cache_config)
        assert kv_cache_config is not None
        assert kv_cache_config.kv_cache_layout is not None
        assert vllm_config.kv_transfer_config is not None
        model = vllm_config.model_config
        self.hidden_layer = model.hf_config.talker_config.accept_hidden_layer
        self.hidden_size = model.hf_text_config.hidden_size
        self.hs_dtype = model.dtype
        group = kv_cache_config.kv_cache_groups[0]
        spec = group.kv_cache_spec
        self.block_size = spec.block_size
        self._row_bytes = 2 * self.hidden_size * torch.empty((), dtype=self.hs_dtype).element_size()
        self._hs_block_bytes = self.block_size * self._row_bytes + 448
        num_layers = len(getattr(group, "layer_names", ())) or getattr(model.hf_text_config, "num_hidden_layers", 48)
        page_bytes = getattr(spec, "page_size_bytes", 0) or (self.block_size * 2 * 4 * 128 * 2)
        self._kv_block_bytes = int(page_bytes) * int(num_layers)
        self.namespace = cache_namespace(
            self.settings, model.hf_config, self.hidden_layer, self.hs_dtype, spec, kv_cache_config.kv_cache_layout
        )
        # Isolate native KV as well as HS across incompatible checkpoints.
        # Asynchronous lookup keeps a slow Store off the scheduling path.
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        extra.update(cache_prefix=self.namespace, load_async=True, lookup_async=True, save_decode_cache=False)
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self._capacity_only = False
        self._kv_cache_config = kv_cache_config
        self._kv_cache_events = None
        if role == KVConnectorRole.SCHEDULER:
            if self.settings.enable_costate:
                self.connector_scheduler = CoStateStoreScheduler(vllm_config, kv_cache_config)
                self._costate_dir = UnifiedPrefixDirectory(
                    block_size=self.block_size,
                    kv_block_bytes=self._kv_block_bytes,
                    hs_block_bytes=self._hs_block_bytes,
                    settings=self.settings,
                )
                self._cost_gate = OnlineCostGate(
                    block_size=self.block_size,
                    row_bytes=self._row_bytes,
                    max_hs_bytes=self.settings.max_hs_bytes,
                    max_pending_transfers=self.settings.max_pending_transfers,
                    header_allowance=_HS_HEADER_ALLOWANCE,
                )
            else:
                self.connector_scheduler = MooncakeStoreScheduler(vllm_config, kv_cache_config)
            self._require_ordered_async_lookup(self.connector_scheduler)
        else:
            if self.settings.enable_costate:
                self._host_hs_cache = HostHSCache(self.settings.host_hs_cache_bytes)
            # The pinned upstream constructor reads a process environment
            # variable. Keep this scoped to construction and restore it.
            with _INIT_LOCK:
                previous = os.environ.get("MOONCAKE_CONFIG_PATH")
                os.environ["MOONCAKE_CONFIG_PATH"] = self.settings.config_file
                try:
                    self.connector_worker = JointStoreWorker(
                        vllm_config,
                        kv_cache_config,
                        self.namespace,
                        self.settings.max_hs_bytes // (3 * self._row_bytes),
                        enable_costate=self.settings.enable_costate,
                    )
                finally:
                    if previous is None:
                        os.environ.pop("MOONCAKE_CONFIG_PATH", None)
                    else:
                        os.environ["MOONCAKE_CONFIG_PATH"] = previous
            # Loads have running deadlines; a save waiting for its native KV
            # object must never hold the worker a load needs.
            self._read_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omni-mooncake-hs-read")
            self._save_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="omni-mooncake-hs-save")

    @staticmethod
    def _require_ordered_async_lookup(scheduler: MooncakeStoreScheduler) -> None:
        """Fail at startup if the pinned lookup client contract has changed.

        The stall guard queues a marker job behind pending lookups and treats
        its completion as proof that they have ended. That holds only while
        the client runs lookups one at a time, in submission order.
        """
        client = scheduler.client
        executor = getattr(client, "executor", None)
        if (
            scheduler.lookup_async is not True
            or not isinstance(executor, ThreadPoolExecutor)
            or executor._max_workers != 1
            or not isinstance(getattr(client, "futures", None), dict)
            or not callable(getattr(client, "discard", None))
        ):
            raise ValueError(
                "Mooncake KV/HS requires the pinned vLLM lookup client, which runs "
                "asynchronous lookups one at a time; recheck the lookup stall guard "
                "before upgrading vLLM"
            )

    @staticmethod
    def _validate_supported(config: VllmConfig, kv_config: KVCacheConfig | None) -> None:
        model = config.model_config
        parallel = config.parallel_config
        if not current_platform.is_cuda() or not config.use_v2_model_runner:
            raise ValueError("Mooncake KV/HS requires CUDA and Model Runner V2")
        if any(
            size != 1
            for size in (
                parallel.tensor_parallel_size,
                parallel.pipeline_parallel_size,
                parallel.data_parallel_size,
                parallel.decode_context_parallel_size,
                parallel.prefill_context_parallel_size,
            )
        ):
            raise ValueError("Mooncake KV/HS requires a single TP=1 serving replica")
        if getattr(model, "model_stage", None) != "thinker" or model.hf_config.model_type != "qwen3_omni_moe":
            raise ValueError("Mooncake KV/HS supports the Qwen3-Omni Thinker only")
        if model.quantization is not None or config.quant_config is not None:
            raise ValueError("Mooncake KV/HS does not yet support a quantized Thinker")
        extra_mc = (
            config.kv_transfer_config.kv_connector_extra_config.get("omni_mooncake", {})
            if config.kv_transfer_config is not None
            and isinstance(config.kv_transfer_config.kv_connector_extra_config, dict)
            else {}
        )
        allow_local_prefix = bool(
            isinstance(extra_mc, dict) and extra_mc.get("enable_costate") and extra_mc.get("enable_local_tier")
        )
        if (
            (config.cache_config.enable_prefix_caching and not allow_local_prefix)
            or config.speculative_config
            or config.lora_config
        ):
            raise ValueError("Mooncake KV/HS does not support local prefix caching, speculation, or LoRA")
        if getattr(model, "session_mode", "turn") not in ("turn", "turn_based"):
            raise ValueError("Mooncake KV/HS duplex serving is not yet supported")
        transfer = config.kv_transfer_config
        if transfer is None or transfer.kv_load_failure_policy != "recompute":
            raise ValueError("Mooncake KV/HS requires kv_load_failure_policy='recompute'")
        if transfer.kv_role != "kv_both":
            raise ValueError("Mooncake KV/HS requires kv_role='kv_both'")
        if getattr(model, "engine_output_type", "text") == "text":
            raise ValueError("Mooncake KV/HS requires Thinker capture outputs for Talker")
        if kv_config is None or len(kv_config.kv_cache_groups) != 1:
            raise ValueError("Mooncake KV/HS requires one full-attention KV group")
        if type(kv_config.kv_cache_groups[0].kv_cache_spec) is not FullAttentionSpec:
            raise ValueError("Mooncake KV/HS requires ordinary full attention")
        if kv_config.kv_cache_layout is None:
            raise ValueError("Mooncake KV/HS requires a resolved physical KV cache layout")

    def cap_local_prefix_cache_hit(self, request: Request, max_cache_hit_length: int | None) -> int | None:
        if not self.settings.enable_costate or not self.settings.enable_local_tier or self._costate_dir is None:
            return max_cache_hit_length
        prompt_limit = max(0, (request.num_prompt_tokens - 1) // self.block_size * self.block_size)
        full_blocks = request.num_prompt_tokens // self.block_size
        prompt_hashes = [bytes(h) for h in request.block_hashes[:full_blocks]]
        text_only = is_text_only_request(request) if self.settings.enable_modality_decouple else False
        self._costate_dir.record_request_visit(request.request_id, prompt_hashes, is_text_only=text_only)
        cap = prompt_limit if text_only else min(prompt_limit, self._costate_dir.contiguous_hs_tokens(prompt_hashes))
        if max_cache_hit_length is not None:
            cap = min(cap, max_cache_hit_length)
        return cap

    def get_num_new_matched_tokens(self, request: Request, num_computed_tokens: int) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        req_id = request.request_id
        if req_id in self._bypass or self._lookup_stalled():
            self._end_lookup(req_id)
            return 0, False
        if self.settings.enable_costate and self._costate_dir is not None:
            full_blocks = request.num_prompt_tokens // self.block_size
            prompt_hashes = [bytes(h) for h in request.block_hashes[:full_blocks]]
            text_only = is_text_only_request(request) if self.settings.enable_modality_decouple else False
            self._costate_dir.record_request_visit(req_id, prompt_hashes, is_text_only=text_only)
            prompt_limit = max(0, (request.num_prompt_tokens - 1) // self.block_size * self.block_size)
            if num_computed_tokens >= prompt_limit:
                self._end_lookup(req_id)
                return 0, False
        started = self._lookup_started.setdefault(req_id, time.monotonic())
        matched, load_async = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if matched is not None:
            self._lookup_started.pop(req_id, None)
            return self._cap_at_prompt(request, num_computed_tokens, matched, load_async)
        if time.monotonic() - started < self.settings.load_timeout_seconds:
            return None, False
        logger.warning(
            "Mooncake prefix lookup for %s exceeded %.1f s; computing the request without the cache",
            req_id,
            self.settings.load_timeout_seconds,
        )
        # The pinned client sends lookups one at a time over one socket, so a
        # stalled reply delays every later lookup. Treat new requests as
        # misses until the work queued before this marker has finished.
        self._lookup_probe = self.connector_scheduler.client.executor.submit(_lookup_queue_drained)
        self._bypass.add(req_id)
        self._end_lookup(req_id)
        return 0, False

    def _lookup_stalled(self) -> bool:
        if self._lookup_probe is not None and self._lookup_probe.done():
            self._lookup_probe = None
        return self._lookup_probe is not None

    def _end_lookup(self, req_id: str) -> None:
        assert self.connector_scheduler is not None
        # Cancels a queued lookup; a running one finishes unobserved.
        self.connector_scheduler.client.discard(req_id)
        self.connector_scheduler.load_specs.pop(req_id, None)
        self._lookup_started.pop(req_id, None)

    def _cap_at_prompt(
        self, request: Request, num_computed_tokens: int, matched: int, load_async: bool
    ) -> tuple[int, bool]:
        """Keep an admitted hit within the prompt blocks whose HS can be restored.

        A preempted request resumes with its generated tokens, and the pinned
        lookup covers those too, so a longer stored prefix can match past the
        prompt. Those positions are recomputed instead.
        """
        assert self.connector_scheduler is not None
        if self.settings.enable_costate and self._costate_dir is not None and self._cost_gate is not None:
            spec = self.connector_scheduler.load_specs.get(request.request_id)
            if spec is None:
                return 0, False
            kv_hit, joint_hit, clean_boundaries = split_costate_lookup_result(
                MooncakeLookupResult(spec.kvpool_cached_tokens, spec.tail_key_boundaries)
            )
            full_blocks = request.num_prompt_tokens // self.block_size
            prompt_hashes = [bytes(h) for h in request.block_hashes[:full_blocks]]
            self._costate_dir.observe_lookup(prompt_hashes, kv_hit, joint_hit)
            text_only = is_text_only_request(request) if self.settings.enable_modality_decouple else False
            limit = max(0, (request.num_prompt_tokens - 1) // self.block_size * self.block_size)
            candidate_total = min(limit, kv_hit if text_only else joint_hit)
            if candidate_total <= num_computed_tokens:
                del self.connector_scheduler.load_specs[request.request_id]
                return 0, False
            host_hs_ok = self._costate_dir.host_hs_covers_prefix(prompt_hashes, num_computed_tokens)
            active_prefills = sum(1 for p in self._active_req_plans.values() if p.ext_load_tokens == 0)
            chosen_total = self._cost_gate.select_external_tokens(
                local_tokens=num_computed_tokens,
                candidate_total_tokens=candidate_total,
                is_text_only=text_only,
                host_hs_covers_local=host_hs_ok,
                enable_cost_gate=self.settings.enable_cost_gate,
                active_gpu_prefills=active_prefills,
            )
            chosen_total -= chosen_total % self.block_size
            if chosen_total <= num_computed_tokens:
                del self.connector_scheduler.load_specs[request.request_id]
                return 0, False
            spec.kvpool_cached_tokens = chosen_total
            spec.tail_key_boundaries = tuple(replace(b, num_tokens=chosen_total) for b in clean_boundaries)
            hs_needed = 0 if text_only else ((chosen_total - num_computed_tokens) if host_hs_ok else chosen_total)
            self._cost_gate.reserve_read(request.request_id, hs_needed)
            return chosen_total - num_computed_tokens, load_async

        spec = self.connector_scheduler.load_specs.get(request.request_id)
        limit = request.num_prompt_tokens - request.num_prompt_tokens % self.block_size
        if spec is None or spec.kvpool_cached_tokens <= limit:
            return matched, load_async
        if limit <= num_computed_tokens:
            del self.connector_scheduler.load_specs[request.request_id]
            return 0, False
        spec.kvpool_cached_tokens = limit
        spec.tail_key_boundaries = tuple(replace(boundary, num_tokens=limit) for boundary in spec.tail_key_boundaries)
        return limit - num_computed_tokens, load_async

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: Any,
        num_external_tokens: int,
    ) -> None:
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if self.settings.enable_costate and self._cost_gate is not None:
            if num_external_tokens <= 0:
                self._cost_gate.release_read(request.request_id)
            else:
                self._ext_loaded_reqs.add(request.request_id)

    def reset_cache(self) -> bool | None:
        if self.role != KVConnectorRole.SCHEDULER:
            return None
        # The pinned reset drains native KV writes and then removes every
        # object in the Store. HS writes are outside that barrier, so an old
        # save could repopulate the cache, and other namespaces would be lost.
        logger.warning("Mooncake KV/HS does not support online connector reset; use a new namespace instead")
        return False

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        super().update_connector_output(connector_output)
        ext_loaded = getattr(self, "_ext_loaded_reqs", None)
        for req_id in connector_output.failed_recving or ():
            self._bypass.add(req_id)
            if ext_loaded is not None:
                ext_loaded.discard(req_id)
            if self._cost_gate is not None:
                self._cost_gate.release_read(req_id)
        for req_id in connector_output.finished_recving or ():
            if self._cost_gate is not None:
                self._cost_gate.release_read(req_id)
            if req_id not in self._bypass:
                continue
            # Rotate only after native copies drain and recovery can run.
            self._scheduler_attempts.pop(req_id, None)
        if getattr(self.settings, "enable_costate", False) and self._costate_dir is not None and self._cost_gate is not None:
            wmeta = connector_output.kv_connector_worker_meta
            if isinstance(wmeta, CoStateWorkerMetadata):
                self._costate_dir.reconcile_worker_meta(wmeta)
                self._cost_gate.update_ema(wmeta.obs_ext_kv_ms_per_block, wmeta.obs_ext_hs_ms_per_block)

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        if request.status in (RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR):
            self._cancelled.add(request.request_id)
        return super().request_finished_all_groups(request, block_ids)

    def _prepare_costate_step_plans(self, scheduler_output: SchedulerOutput) -> dict[str, CoStateReqPlan]:
        assert isinstance(self.connector_scheduler, CoStateStoreScheduler)
        assert self._costate_dir is not None and self._cost_gate is not None
        step_plans: dict[str, CoStateReqPlan] = {}
        new_ids = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        cached_ids = set(scheduler_output.scheduled_cached_reqs.req_ids)

        # 1. Pending external loads not yet scheduled for compute
        for req_id, lspec in self.connector_scheduler.load_specs.items():
            if req_id in new_ids or req_id in cached_ids or not lspec.can_load:
                continue
            req_tuple = self.connector_scheduler._unfinished_requests.get(req_id)
            if req_tuple is None:
                continue
            ureq = req_tuple[0]
            plen = ureq.num_prompt_tokens
            phashes = [bytes(h) for h in ureq.block_hashes[: plen // self.block_size]]
            tonly = is_text_only_request(ureq) if self.settings.enable_modality_decouple else False
            step_plans[req_id] = CoStateReqPlan(
                req_id=req_id,
                prompt_len=plen,
                block_hashes=phashes,
                is_text_only=tonly,
                local_kv_tokens=lspec.vllm_cached_tokens,
                ext_load_tokens=lspec.kvpool_cached_tokens - lspec.vllm_cached_tokens,
                ext_load_hs=not tonly,
            )

        # 2. Newly scheduled compute requests
        for new_req in scheduler_output.scheduled_new_reqs:
            rid = new_req.req_id
            req_tuple = self.connector_scheduler._unfinished_requests.get(rid)
            if req_tuple is None:
                continue
            ureq = req_tuple[0]
            plen = ureq.num_prompt_tokens
            phashes = [bytes(h) for h in ureq.block_hashes[: plen // self.block_size]]
            tonly = is_text_only_request(ureq) if self.settings.enable_modality_decouple else False
            self._costate_dir.record_request_visit(rid, phashes, is_text_only=tonly)
            was_ext = rid in self._ext_loaded_reqs
            self._ext_loaded_reqs.discard(rid)
            local_kv_tok = 0 if was_ext else new_req.num_computed_tokens
            avail_hs = self._cost_gate.available_hs_save_blocks()
            save_kv, save_hs, evicts = self._costate_dir.plan_store_write_and_evictions(
                block_hashes=phashes,
                target_tokens=(plen // self.block_size) * self.block_size,
                is_text_only=tonly,
                avail_hs_blocks=avail_hs,
            )
            self._pending_evict_hashes.extend(evicts)
            promote_tok = min(local_kv_tok, save_hs) if (local_kv_tok > 0 and save_hs > 0) else 0
            if save_hs > 0:
                self._cost_gate.reserve_save(rid, save_hs)
            plan = CoStateReqPlan(
                req_id=rid,
                prompt_len=plen,
                block_hashes=phashes,
                is_text_only=tonly,
                local_kv_tokens=local_kv_tok,
                ext_load_tokens=0,
                ext_load_hs=not tonly,
                save_store_kv_tokens=save_kv,
                save_store_hs_tokens=save_hs,
                save_host_hs=bool(self.settings.enable_local_tier and not tonly),
                promote_from_host_hs_tokens=promote_tok,
            )
            self._active_req_plans[rid] = plan
            step_plans[rid] = plan
            if plan.save_host_hs:
                done_tok = min(plen, new_req.num_computed_tokens + scheduler_output.num_scheduled_tokens.get(rid, 0))
                self._costate_dir.record_shadow_host_hs(phashes[: done_tok // self.block_size])
                if done_tok >= plen:
                    self._cost_gate.release_save(rid)

        # 3. Cached / chunked-prefill / resumed requests
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, rid in enumerate(cached_reqs.req_ids):
            if rid in cached_reqs.resumed_req_ids:
                req_tuple = self.connector_scheduler._unfinished_requests.get(rid)
                if req_tuple is None:
                    continue
                ureq = req_tuple[0]
                plen = ureq.num_prompt_tokens
                phashes = [bytes(h) for h in ureq.block_hashes[: plen // self.block_size]]
                tonly = is_text_only_request(ureq) if self.settings.enable_modality_decouple else False
                was_ext = rid in self._ext_loaded_reqs
                self._ext_loaded_reqs.discard(rid)
                local_kv_tok = 0 if was_ext else ureq.num_computed_tokens
                avail_hs = self._cost_gate.available_hs_save_blocks()
                save_kv, save_hs, evicts = self._costate_dir.plan_store_write_and_evictions(
                    block_hashes=phashes,
                    target_tokens=(plen // self.block_size) * self.block_size,
                    is_text_only=tonly,
                    avail_hs_blocks=avail_hs,
                )
                self._pending_evict_hashes.extend(evicts)
                plan = CoStateReqPlan(
                    req_id=rid,
                    prompt_len=plen,
                    block_hashes=phashes,
                    is_text_only=tonly,
                    local_kv_tokens=local_kv_tok,
                    ext_load_tokens=0,
                    ext_load_hs=not tonly,
                    save_store_kv_tokens=save_kv,
                    save_store_hs_tokens=save_hs,
                    save_host_hs=bool(self.settings.enable_local_tier and not tonly),
                    promote_from_host_hs_tokens=min(local_kv_tok, save_hs) if (local_kv_tok > 0 and save_hs > 0) else 0,
                )
                self._active_req_plans[rid] = plan
                step_plans[rid] = plan
            elif (plan := self._active_req_plans.get(rid)) is not None:
                step_plans[rid] = plan
                if plan.save_host_hs:
                    cur_comp = cached_reqs.num_computed_tokens[idx]
                    done_tok = min(plan.prompt_len, cur_comp + scheduler_output.num_scheduled_tokens.get(rid, 0))
                    self._costate_dir.record_shadow_host_hs(plan.block_hashes[: done_tok // self.block_size])
                    if done_tok >= plan.prompt_len:
                        self._cost_gate.release_save(rid)

        self.connector_scheduler.save_kv_limits = {
            rid: p.save_store_kv_tokens for rid, p in self._active_req_plans.items()
        }
        return step_plans

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> JointMetadata:
        retired = set(scheduler_output.finished_req_ids) | set(scheduler_output.preempted_req_ids or ())
        active_plans = getattr(self, "_active_req_plans", None)
        ext_loaded = getattr(self, "_ext_loaded_reqs", None)
        for req_id in retired:
            self._scheduler_attempts.pop(req_id, None)
            self._lookup_started.pop(req_id, None)
            self._bypass.discard(req_id)
            if active_plans is not None:
                active_plans.pop(req_id, None)
            if ext_loaded is not None:
                ext_loaded.discard(req_id)
            if self._cost_gate is not None:
                self._cost_gate.release_read(req_id)
                self._cost_gate.release_save(req_id)
        if self._costate_dir is not None:
            for req_id in scheduler_output.finished_req_ids:
                self._costate_dir.retire_request(req_id)

        step_plans: dict[str, CoStateReqPlan] = {}
        evict_batch: list[bytes] = []
        if getattr(self.settings, "enable_costate", False) and isinstance(self.connector_scheduler, CoStateStoreScheduler):
            step_plans = self._prepare_costate_step_plans(scheduler_output)
            if self._pending_evict_hashes:
                evict_batch = list(dict.fromkeys(self._pending_evict_hashes))
                self._pending_evict_hashes.clear()

        native = cast(MooncakeStoreConnectorMetadata, super().build_connector_meta(scheduler_output))
        req_ids = (
            {req.req_id for req in native.requests}
            | set(scheduler_output.num_scheduled_tokens)
            | set(step_plans)
        )
        for req_id in req_ids:
            if req_id not in self._scheduler_attempts:
                self._scheduler_attempts[req_id] = uuid.uuid4().hex
        metadata = JointMetadata(
            native,
            dict(self._scheduler_attempts),
            set(scheduler_output.finished_req_ids),
            set(self._cancelled) | set(scheduler_output.preempted_req_ids or ()),
            req_plans=step_plans,
            evict_hashes=evict_batch,
        )
        self._cancelled.clear()
        return metadata

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        base = super().build_connector_worker_meta()
        if not self.settings.enable_costate:
            return base
        with self._lock:
            pending = self._pending_worker_meta
            self._pending_worker_meta = CoStateWorkerMetadata()
        if isinstance(base, MooncakeStoreWorkerMetadata) and base.completed_saves:
            pending.completed_saves.update(base.completed_saves)
        return pending if pending.has_payload() else None

    def _evict_store_blocks(self, block_hashes: list[bytes]) -> None:
        if self.connector_worker is None or not block_hashes:
            return
        store = self.connector_worker.store
        with self._lock:
            active_hashes = {
                bh
                for p in self._reads.values()
                for bh in p.attempt.hashes
            }
        evicted: list[bytes] = []
        for raw_hash in block_hashes:
            bh = bytes(raw_hash)
            if bh in active_hashes:
                continue
            hkey = hs_key(self.namespace, bh)
            kv_key = self.connector_worker.token_dbs[0].key_for(bh)
            try:
                store.remove(hkey + ":ready", force=True)
                store.remove(hkey, force=True)
                store.remove(kv_key, force=True)
                evicted.append(bh)
            except (RuntimeError, OSError) as exc:
                logger.warning("CoState store eviction skipped for block: %s", exc)
        if evicted:
            with self._lock:
                self._pending_worker_meta.store_evicted.extend(evicted)

    def _promote_host_hs_prefix(self, attempt: Attempt, num_tokens: int) -> None:
        if self._host_hs_cache is None or num_tokens <= 0:
            return
        num_blocks = num_tokens // self.block_size
        if num_blocks <= 0 or num_blocks > len(attempt.hashes):
            return
        blocks: list[tuple[int, int, CaptureTensors]] = []
        for idx in range(num_blocks):
            tensors = self._host_hs_cache.get_block(attempt.hashes[idx])
            if tensors is None:
                break
            blocks.append((idx * self.block_size, (idx + 1) * self.block_size, tensors))
        if not blocks:
            return
        reserve = (
            3 * sum(t.numel() * t.element_size() for _, _, group in blocks for t in group.values())
            + _HS_HEADER_ALLOWANCE
        )
        if (
            len(self._saves) < self.settings.max_pending_transfers
            and self._memory_used() + reserve <= self.settings.max_hs_bytes
        ):
            self._reserved_bytes += reserve
            save = PendingSave(attempt, blocks, reserve)
            previous = next(
                (future for future, pending in reversed(self._saves.items()) if pending.attempt is attempt),
                None,
            )
            self._queue_save(save, previous)

    def bind_connector_metadata(self, connector_metadata: KVConnectorMetadata) -> None:
        # Bind before the forward: native async load submission itself may be
        # deferred until post_forward, after the runner snapshots the attempt.
        connector_metadata = cast(JointMetadata, connector_metadata)
        metadata = copy(connector_metadata)
        evict_hashes = list(connector_metadata.omni_evict_hashes) if connector_metadata.omni_evict_hashes else []
        with self._lock:
            cancelled_tokens = set()
            for req_id in connector_metadata.omni_finished:
                self._delivered_until.pop(req_id, None)
            for req_id in connector_metadata.omni_finished | connector_metadata.preempted_req_ids:
                old = self._attempts.pop(req_id, None)
                if old:
                    # A normal response may finish before its best-effort
                    # save drains. Only abort/preemption invalidates writes.
                    old.cancelled = req_id in connector_metadata.omni_cancelled
                    old.rows = None
                    self._retired.add(old.token)
                    if old.cancelled:
                        cancelled_tokens.add(old.token)
            for req_id, token in connector_metadata.omni_attempts.items():
                old = self._attempts.get(req_id)
                if old is not None and old.token != token:
                    old.cancelled = True
                    old.rows = None
                    self._retired.add(old.token)
                    cancelled_tokens.add(old.token)
                if old is None or old.token != token:
                    self._attempts[req_id] = Attempt(req_id, token, 0)
            for req_id, plan in connector_metadata.omni_req_plans.items():
                attempt = self._attempts.get(req_id)
                if attempt is None:
                    continue
                attempt.prompt_len = plan.prompt_len
                attempt.hashes = list(plan.block_hashes)
                attempt.is_text_only = plan.is_text_only
                attempt.local_kv_tokens = plan.local_kv_tokens
                attempt.ext_load_tokens = plan.ext_load_tokens
                attempt.ext_load_hs = plan.ext_load_hs
                attempt.save_store_hs_tokens = plan.save_store_hs_tokens
                attempt.save_host_hs = plan.save_host_hs
                attempt.promote_from_host_hs_tokens = plan.promote_from_host_hs_tokens
                if plan.promote_from_host_hs_tokens > 0 and not attempt.promoted_host_hs:
                    attempt.promoted_host_hs = True
                    self._promote_host_hs_prefix(attempt, plan.promote_from_host_hs_tokens)
            metadata.requests = []
            for req in connector_metadata.requests:
                attempt = self._attempts[req.req_id]
                if req.num_prompt_tokens is not None:
                    attempt.prompt_len = req.num_prompt_tokens
                if req.block_hashes:
                    attempt.hashes = list(req.block_hashes)
                native = copy(req)
                native.req_id = attempt.token
                metadata.requests.append(native)
            # Native preemption immediately invalidates queued store jobs.
            # Normal finishes instead retire through get_transfer_results,
            # which preserves their ledger until all pinned jobs drain.
            metadata.preempted_req_ids = cancelled_tokens
            metadata.unfinished_request_ids = {a.token for a in self._attempts.values()}
        if evict_hashes and self._save_executor is not None:
            self._save_executor.submit(self._evict_store_blocks, evict_hashes)
        super().bind_connector_metadata(metadata)

    def current_attempt(self, req_id: str) -> str | None:
        with self._lock:
            attempt = self._attempts.get(req_id)
            return attempt.token if attempt is not None and not attempt.cancelled else None

    def _spec(self, block_hash: bytes, index: int) -> HSBlockSpec:
        return HSBlockSpec(
            self.namespace,
            prefix_digest(block_hash),
            index * self.block_size,
            (index + 1) * self.block_size,
            self.hidden_layer,
            self.hidden_size,
            self.hs_dtype,
        )

    def _read_hs_range(
        self,
        attempt: Attempt,
        start_tokens: int,
        end_tokens: int,
        prefix_tensors: CaptureTensors | None = None,
    ) -> dict[str, torch.Tensor]:
        assert self.connector_worker is not None
        store = self.connector_worker.store
        if end_tokens > attempt.prompt_len:
            raise ValueError("HS prefix extends past the prompt")
        start_blk = start_tokens // self.block_size
        end_blk = end_tokens // self.block_size
        hashes = attempt.hashes[start_blk:end_blk]
        if end_blk > start_blk and len(hashes) != (end_blk - start_blk):
            raise ValueError("incomplete HS prefix")
        t0 = time.perf_counter()
        blocks: list[CaptureTensors] = []
        if prefix_tensors is not None:
            blocks.append(prefix_tensors)
        per_batch = max(1, _READ_BATCH_BYTES // (self.block_size * self._row_bytes))
        added_host: list[bytes] = []
        evicted_host: list[bytes] = []
        for first in range(0, len(hashes), per_batch):
            if attempt.cancelled:
                raise ValueError("cancelled HS load")
            batch = hashes[first : first + per_batch]
            keys = [hs_key(self.namespace, block_hash) for block_hash in batch]
            values = store.get_batch([name for key in keys for name in (key, key + ":ready")])
            if len(values) != 2 * len(keys):
                raise ValueError("HS batch read failed")
            for offset, block_hash in enumerate(batch):
                payload, ready = values[2 * offset], values[2 * offset + 1]
                if not isinstance(payload, bytes) or not payload:
                    raise ValueError("missing HS object")
                if ready != hashlib.sha256(payload).digest():
                    raise ValueError("HS object has no matching completion record")
                decoded = decode_hs(payload, self._spec(block_hash, start_blk + first + offset))
                blocks.append(decoded)
                if self._host_hs_cache is not None and self.settings.enable_local_tier:
                    ok, ev = self._host_hs_cache.put_block(bytes(block_hash), decoded)
                    if ok:
                        added_host.append(bytes(block_hash))
                    evicted_host.extend(ev)
            del values
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ext_count = end_blk - start_blk
        if self.settings.enable_costate and (ext_count > 0 or added_host or evicted_host):
            with self._lock:
                if ext_count > 0:
                    self._pending_worker_meta.obs_ext_hs_ms_per_block.append(elapsed_ms / ext_count)
                self._pending_worker_meta.host_hs_added.extend(added_host)
                self._pending_worker_meta.host_hs_evicted.extend(evicted_host)
        if not blocks:
            raise ValueError("empty HS restore")
        return {name: torch.cat([block[name] for block in blocks]) for name in ("embedding", "hidden")}

    def _read_hs(self, attempt: Attempt, tokens: int) -> dict[str, torch.Tensor]:
        return self._read_hs_range(attempt, 0, tokens, prefix_tensors=None)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        metadata = cast(JointMetadata, self._get_connector_metadata())
        now = time.monotonic()
        with self._lock:
            by_token = {a.token: a for a in self._attempts.values()}
            for req in metadata.requests:
                spec = req.load_spec
                if spec is None or not spec.can_load:
                    continue
                attempt = by_token[req.req_id]
                tokens = spec.kvpool_cached_tokens
                local_tokens = spec.vllm_cached_tokens
                ext_kv_blocks = max(0, (tokens - local_tokens) // self.block_size)

                # CoState Mechanism 1A: Text-only requests load KV only (0 HS cost)
                if self.settings.enable_costate and (attempt.is_text_only or not attempt.ext_load_hs):
                    pending = PendingRead(
                        attempt,
                        0,
                        now + self.settings.load_timeout_seconds,
                        hs_needed=False,
                        started_at=time.perf_counter(),
                        ext_kv_blocks=ext_kv_blocks,
                        ext_hs_blocks=0,
                    )
                    pending.block_ids = frozenset(req.block_ids[0][: tokens // self.block_size])
                    self._reads[attempt.token] = pending
                    continue

                # CoState Mechanism 1B: Reuse HostHSCache for [0 : local_tokens) when available
                prefix_tensors: CaptureTensors | None = None
                hs_start_tokens = 0
                if (
                    self.settings.enable_costate
                    and self.settings.enable_local_tier
                    and self._host_hs_cache is not None
                    and local_tokens > 0
                ):
                    prefix_tensors = self._host_hs_cache.assemble_prefix(
                        attempt.hashes[: local_tokens // self.block_size]
                    )
                    if prefix_tensors is not None:
                        hs_start_tokens = local_tokens

                hs_fetch_tokens = tokens - hs_start_tokens
                reserve = 3 * hs_fetch_tokens * self._row_bytes + _HS_HEADER_ALLOWANCE
                pending = PendingRead(
                    attempt,
                    reserve,
                    now + self.settings.load_timeout_seconds,
                    hs_needed=True,
                    started_at=time.perf_counter(),
                    ext_kv_blocks=ext_kv_blocks,
                    ext_hs_blocks=hs_fetch_tokens // self.block_size,
                )
                pending.block_ids = frozenset(req.block_ids[0][: tokens // self.block_size])
                # Saves have separate slots and workers, so they cannot
                # prevent a load from being admitted or started.
                if (
                    len(self._reads) >= self.settings.max_pending_transfers
                    or self._memory_used() + reserve > self.settings.max_hs_bytes
                ):
                    pending.failed = True
                    pending.reserved_bytes = 0
                else:
                    self._reserved_bytes += reserve
                    assert self._read_executor is not None
                    if hs_start_tokens > 0 and prefix_tensors is not None:
                        pending.future = self._read_executor.submit(
                            self._read_hs_range, attempt, hs_start_tokens, tokens, prefix_tensors
                        )
                    else:
                        pending.future = self._read_executor.submit(self._read_hs, attempt, tokens)
                self._reads[attempt.token] = pending
        # Even an HS admission failure drains the native KV operation before
        # allowing the scheduler to free its destination blocks.
        super().start_load_kv(forward_context, **kwargs)

    def _memory_used(self) -> int:
        return self._reserved_bytes + sum(a.rows.buffered_bytes for a in self._attempts.values() if a.rows)

    def get_transfer_results(self, finished_req_ids: set[str]) -> KVConnectorTransferResults:
        with self._lock:
            retired = set(self._retired)
            self._retired.clear()
        native = super().get_transfer_results(retired)
        failed_blocks = super().get_block_ids_with_load_errors()
        finished, failed = set(), set()
        with self._lock:
            for token in native.failed_recving:
                if token in self._reads:
                    self._reads[token].failed = True
            for token in native.finished_recving:
                if token in self._reads:
                    pending_item = self._reads[token]
                    pending_item.kv_done = True
                    if self.settings.enable_costate and pending_item.ext_kv_blocks > 0 and pending_item.started_at > 0:
                        kv_ms = (time.perf_counter() - pending_item.started_at) * 1000.0
                        self._pending_worker_meta.obs_ext_kv_ms_per_block.append(kv_ms / pending_item.ext_kv_blocks)
            for token, pending in list(self._reads.items()):
                attempt = pending.attempt
                if pending.block_ids & failed_blocks:
                    pending.failed = True
                if time.monotonic() >= pending.deadline:
                    pending.failed = True
                hs_done = (not pending.hs_needed) or pending.future is None or pending.future.done()
                tensors = None
                if (
                    pending.hs_needed
                    and not pending.failed
                    and not attempt.cancelled
                    and pending.future is not None
                    and hs_done
                ):
                    try:
                        tensors = pending.future.result()
                    except (ValueError, RuntimeError, OSError) as exc:
                        logger.warning("Mooncake HS read failed for %s: %s", attempt.req_id, exc)
                        pending.failed = True
                if pending.failed and not pending.failure_reported and not attempt.cancelled:
                    failed.add(attempt.req_id)
                    pending.failure_reported = True
                if (pending.failed or attempt.cancelled) and pending.kv_done and not pending.completion_reported:
                    # Once the native copy drains, the scheduler may recycle
                    # the destination GPU blocks. HS reads touch CPU buffers
                    # only: a failed or cancelled read may still be running,
                    # so its reservation stays until the future finishes.
                    current = self._attempts.get(attempt.req_id)
                    if current is None or current.token == token:
                        finished.add(attempt.req_id)
                    pending.completion_reported = True
                if not pending.kv_done or not hs_done:
                    continue
                self._reserved_bytes -= pending.reserved_bytes
                if not pending.failed and not attempt.cancelled and pending.hs_needed:
                    assert tensors is not None
                    max_req_hs = (
                        max(self.settings.max_hs_bytes, 3 * attempt.prompt_len * self._row_bytes)
                        if self.settings.enable_costate
                        else self.settings.max_hs_bytes
                    )
                    attempt.rows = RequestHS(attempt.prompt_len, self.block_size, max_req_hs)
                    attempt.rows.restore(tensors)
                current = self._attempts.get(attempt.req_id)
                if not pending.completion_reported and (current is None or current.token == token):
                    finished.add(attempt.req_id)
                del self._reads[token]
            for future, save in list(self._saves.items()):
                if future.done():
                    try:
                        future.result()
                    except (ValueError, RuntimeError, OSError) as exc:
                        logger.warning("Mooncake HS save skipped: %s", exc)
                    self._release_staging(save)
                    del self._saves[future]
        return KVConnectorTransferResults(finished_sending=set(), finished_recving=finished, failed_recving=failed)

    def get_block_ids_with_load_errors(self) -> set[int]:
        # get_transfer_results consumes block-level failures and upgrades them
        # to whole-attempt failure so HS state and retry bypass stay coherent.
        return set()

    def _release_staging(self, save: PendingSave) -> None:
        with self._lock:
            save.blocks.clear()
            self._reserved_bytes -= save.reserved_bytes
            save.reserved_bytes = 0

    def _stop_saving(self, attempt: Attempt) -> None:
        with self._lock:
            if attempt.rows is not None:
                attempt.rows.save_enabled = False

    def _queue_save(self, save: PendingSave, previous: Future[bool] | None) -> None:
        """Count deferred saves against admission without occupying a worker."""
        result: Future[bool] = Future()
        self._saves[result] = save

        def finish(future: Future[bool]) -> None:
            if future.cancelled():
                self._release_staging(save)
                result.set_result(False)
            elif (error := future.exception()) is not None:
                result.set_exception(error)
            else:
                result.set_result(future.result())

        def submit(predecessor: Future[bool] | None) -> None:
            with self._lock:
                if result.done():
                    return
                if save.attempt.cancelled or (
                    predecessor is not None
                    and (predecessor.cancelled() or predecessor.exception() is not None or not predecessor.result())
                ):
                    # Resolve tails first so callbacks see completed results
                    # instead of recursively failing a long successor chain.
                    for future, pending in reversed(self._saves.items()):
                        if pending.attempt is save.attempt and not future.done():
                            self._release_staging(pending)
                            future.set_result(False)
                    return
                assert self._save_executor is not None
                self._save_executor.submit(self._save_hs, save).add_done_callback(finish)

        if previous is None:
            submit(None)
        else:
            previous.add_done_callback(submit)

    def _write_hs_block(
        self, store: Any, attempt: Attempt, block: tuple[int, int, CaptureTensors]
    ) -> tuple[str, str, bytes] | None:
        """Write one HS object; return its KV key, completion key and digest."""
        assert self.connector_worker is not None
        start, _, tensors = block
        index = start // self.block_size
        block_hash = attempt.hashes[index]
        key = hs_key(self.namespace, block_hash)
        spec = self._spec(block_hash, index)
        if store.put(key, encode_hs(spec, tensors)) != 0:
            return None
        # put(existing_key) is immutable in the pinned Store. Read the actual
        # winner so concurrent duplicate writers publish its digest.
        actual = store.get(key)
        if not isinstance(actual, bytes) or not actual:
            return None
        decode_hs(actual, spec)
        kv_key = self.connector_worker.token_dbs[0].key_for(block_hash)
        return kv_key, key + ":ready", hashlib.sha256(actual).digest()

    def _save_hs(self, save: PendingSave) -> bool:
        """Write HS objects, then publish each completion record after its KV.

        Successors are submitted only after this save completes successfully,
        so a failed block stops queued successors and future capture staging.
        Tensors and their staging reservation are released before waiting for
        native KV objects, so a slow KV write holds only this save worker.
        """
        assert self.connector_worker is not None
        attempt = save.attempt
        store = self.connector_worker.store
        try:
            commits: list[tuple[str, str, bytes, bytes]] = []
            complete = True
            # Indexing leaves no tensor reference in this frame for the KV wait.
            for position in range(len(save.blocks)):
                if attempt.cancelled:
                    return False
                start_pos = save.blocks[position][0]
                blk_hash = bytes(attempt.hashes[start_pos // self.block_size])
                written = self._write_hs_block(store, attempt, save.blocks[position])
                if written is None:
                    self._stop_saving(attempt)
                    complete = False
                    break
                commits.append((*written, blk_hash))
            self._release_staging(save)
            # HS serialization and Store writes must not consume the KV wait.
            deadline = time.monotonic() + self.settings.load_timeout_seconds
            # Preserve successful leading blocks even if a later write failed.
            committed_hashes: list[bytes] = []
            while commits and not attempt.cancelled and time.monotonic() < deadline:
                exists = store.batch_is_exist([kv_key for kv_key, _, _, _ in commits])
                ready = 0
                while ready < min(len(commits), len(exists)) and exists[ready] == 1:
                    ready += 1
                for _, ready_key, digest, blk_hash in commits[:ready]:
                    if attempt.cancelled:
                        return False
                    self._publish_ready(store, ready_key, digest)
                    committed_hashes.append(blk_hash)
                del commits[:ready]
                if commits:
                    time.sleep(_KV_POLL_SECONDS)
            if committed_hashes and self.settings.enable_costate:
                with self._lock:
                    self._pending_worker_meta.store_kv_saved.extend(committed_hashes)
                    self._pending_worker_meta.store_hs_saved.extend(committed_hashes)
            if commits:
                self._stop_saving(attempt)
            return complete and not commits and not attempt.cancelled
        except (ValueError, RuntimeError, OSError):
            self._stop_saving(attempt)
            raise
        finally:
            self._release_staging(save)

    @staticmethod
    def _publish_ready(store: Any, ready_key: str, digest: bytes) -> None:
        previous = store.get(ready_key)
        # HS can be evicted independently of its completion record. A refilled
        # object needs a new record; immutable put alone would preserve the
        # old digest and poison every later hit.
        if previous and previous != digest:
            if store.remove(ready_key, force=True) != 0:
                raise RuntimeError("Mooncake HS completion record replacement failed")
        if store.put(ready_key, digest) != 0:
            raise RuntimeError("Mooncake HS completion record write failed")

    def _stage_host_hs_blocks(self, attempt: Attempt, start_token: int, tensors: dict[str, torch.Tensor]) -> None:
        if self._host_hs_cache is None or not attempt.save_host_hs:
            return
        end = min(start_token + tensors["embedding"].shape[0], attempt.prompt_len)
        begin = max(start_token, attempt.host_stage_cursor)
        if end <= begin:
            return
        if begin != attempt.host_stage_cursor + attempt.host_partial_rows:
            attempt.host_partial = None
            attempt.host_partial_rows = 0
            attempt.host_stage_cursor = begin - (begin % self.block_size)
            if begin != attempt.host_stage_cursor:
                return
        added: list[bytes] = []
        evicted: list[bytes] = []
        while begin < end:
            if attempt.host_stage_cursor + self.block_size > attempt.prompt_len:
                break
            if attempt.host_partial is None:
                attempt.host_partial = {
                    name: torch.empty((self.block_size, tensor.shape[1]), dtype=tensor.dtype)
                    for name, tensor in tensors.items()
                }
            count = min(end - begin, self.block_size - attempt.host_partial_rows)
            for name, tensor in tensors.items():
                attempt.host_partial[name][attempt.host_partial_rows : attempt.host_partial_rows + count].copy_(
                    tensor[begin - start_token : begin - start_token + count]
                )
            attempt.host_partial_rows += count
            begin += count
            if attempt.host_partial_rows == self.block_size:
                blk_idx = attempt.host_stage_cursor // self.block_size
                if blk_idx < len(attempt.hashes):
                    bh = bytes(attempt.hashes[blk_idx])
                    ok, ev = self._host_hs_cache.put_block(bh, attempt.host_partial)
                    if ok:
                        added.append(bh)
                    evicted.extend(ev)
                attempt.host_stage_cursor += self.block_size
                attempt.host_partial = None
                attempt.host_partial_rows = 0
        if added or evicted:
            self._pending_worker_meta.host_hs_added.extend(added)
            self._pending_worker_meta.host_hs_evicted.extend(evicted)

    def _restore_local_or_store_hs(self, attempt: Attempt, num_tokens: int) -> CaptureTensors:
        num_blocks = num_tokens // self.block_size
        hashes = attempt.hashes[:num_blocks]
        if self._host_hs_cache is not None:
            assembled = self._host_hs_cache.assemble_prefix(hashes)
            if assembled is not None:
                return assembled
        # Fallback to Mooncake Store if any block was evicted from HostHSCache
        return self._read_hs_range(attempt, 0, num_tokens, prefix_tensors=None)

    def capture_step(
        self, req_id: str, attempt_id: str, start_token: int, prompt_len: int, tensors: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor] | None:
        with self._lock:
            attempt = self._attempts.get(req_id)
            if attempt is None or attempt.cancelled or attempt.token != attempt_id:
                return None

            # CoState Mechanism 1A: Text-only turns terminate at Stage 0 without Talker HS
            if self.settings.enable_costate and self.settings.enable_modality_decouple and attempt.is_text_only:
                return tensors

            if attempt.rows is None:
                max_req_hs = (
                    max(self.settings.max_hs_bytes, 3 * prompt_len * self._row_bytes)
                    if self.settings.enable_costate
                    else self.settings.max_hs_bytes
                )
                attempt.rows = RequestHS(prompt_len, self.block_size, max_req_hs)
                # CoState Mechanism 1B: Supply 8% HS for local GPU KV hits
                if self.settings.enable_costate and attempt.local_kv_tokens > 0 and attempt.ext_load_tokens == 0:
                    local_hs = self._restore_local_or_store_hs(attempt, attempt.local_kv_tokens)
                    attempt.rows.restore(local_hs)
                    attempt.host_stage_cursor = attempt.local_kv_tokens

            if self.settings.enable_costate and attempt.save_store_hs_tokens == 0:
                attempt.rows.save_enabled = False

            # Saving is best effort; output assembly is mandatory. Budget the
            # block copies before RequestHS stages this step's captures.
            fresh_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
            if self._memory_used() + 3 * fresh_bytes > self.settings.max_hs_bytes:
                attempt.rows.save_enabled = False
            merged = attempt.rows.append(start_token, tensors)

            if self.settings.enable_costate and attempt.save_host_hs:
                self._stage_host_hs_blocks(attempt, start_token, tensors)

            blocks = attempt.rows.pop_blocks()
            if self.settings.enable_costate and attempt.save_store_hs_tokens >= 0:
                blocks = [blk for blk in blocks if blk[1] <= attempt.save_store_hs_tokens]
                if attempt.rows._save_cursor >= attempt.save_store_hs_tokens:
                    attempt.rows.save_enabled = False

            reserve = (
                3 * sum(t.numel() * t.element_size() for _, _, group in blocks for t in group.values())
                + _HS_HEADER_ALLOWANCE
            )
            if blocks:
                if (
                    len(self._saves) < self.settings.max_pending_transfers
                    and self._memory_used() + reserve <= self.settings.max_hs_bytes
                ):
                    self._reserved_bytes += reserve
                    save = PendingSave(attempt, blocks, reserve)
                    previous = next(
                        (future for future, pending in reversed(self._saves.items()) if pending.attempt is attempt),
                        None,
                    )
                    self._queue_save(save, previous)
                else:
                    # Lookups stop at the first missing block, so blocks
                    # after a dropped one would be stored but never reused.
                    attempt.rows.save_enabled = False
            # Preemption rebuilds KV/HS, but the downstream accumulator still
            # owns already published rows. Keep its watermark across attempts.
            end = start_token + tensors["embedding"].shape[0]
            merged_start = end - merged["embedding"].shape[0]
            delivered = self._delivered_until.get(req_id, 0)
            trim = min(merged["embedding"].shape[0], max(0, delivered - merged_start))
            if merged_start > delivered:
                raise ValueError("Mooncake handoff would omit prompt rows")
            self._delivered_until[req_id] = max(delivered, end)
            if trim:
                merged = {name: tensor[trim:] for name, tensor in merged.items()}
            return merged

    def audit_store_prefix_chains(self, chains: Sequence[Sequence[bytes]]) -> dict[str, Any]:
        """Directly inspect Mooncake Store keys to compute exact Zombie Byte Ratio."""
        if self.connector_worker is None:
            return {}
        store = self.connector_worker.store
        unique_nodes: dict[bytes, tuple[int, bytes | None]] = {}
        for chain in chains:
            parent: bytes | None = None
            for depth, raw_hash in enumerate(chain):
                bh = bytes(raw_hash)
                if bh not in unique_nodes:
                    unique_nodes[bh] = (depth, parent)
                parent = bh
        if not unique_nodes:
            return {
                "resident_kv_blocks": 0,
                "resident_hs_blocks": 0,
                "resident_total_bytes": 0,
                "zombie_joint_bytes": 0,
                "zombie_mixed_bytes": 0,
                "zbr_joint": 0.0,
                "zbr_mixed": 0.0,
            }
        ordered = sorted(unique_nodes.items(), key=lambda kv: kv[1][0])
        kv_keys = [self.connector_worker.token_dbs[0].key_for(bh) for bh, _ in ordered]
        hs_keys = [name for bh, _ in ordered for name in (hs_key(self.namespace, bh), hs_key(self.namespace, bh) + ":ready")]
        kv_exist = store.batch_is_exist(kv_keys)
        hs_exist = store.batch_is_exist(hs_keys)

        in_kv: dict[bytes, bool] = {}
        in_hs: dict[bytes, bool] = {}
        for idx, (bh, _) in enumerate(ordered):
            in_kv[bh] = bool(idx < len(kv_exist) and kv_exist[idx] == 1)
            in_hs[bh] = bool(2 * idx + 1 < len(hs_exist) and hs_exist[2 * idx] == 1 and hs_exist[2 * idx + 1] == 1)

        valid_text: dict[bytes, bool] = {}
        valid_audio: dict[bytes, bool] = {}
        for bh, (_, parent) in ordered:
            p_text = True if parent is None else valid_text.get(parent, False)
            p_audio = True if parent is None else valid_audio.get(parent, False)
            valid_text[bh] = p_text and in_kv[bh]
            valid_audio[bh] = p_audio and in_kv[bh] and in_hs[bh]

        total_bytes = 0
        zombie_joint = 0
        zombie_mixed = 0
        res_kv = 0
        res_hs = 0
        for bh in unique_nodes:
            if in_kv[bh]:
                res_kv += 1
                total_bytes += self._kv_block_bytes
                if not valid_audio[bh]:
                    zombie_joint += self._kv_block_bytes
                if not valid_text[bh]:
                    zombie_mixed += self._kv_block_bytes
            if in_hs[bh]:
                res_hs += 1
                total_bytes += self._hs_block_bytes
                if not valid_audio[bh]:
                    zombie_joint += self._hs_block_bytes
                    zombie_mixed += self._hs_block_bytes

        return {
            "resident_kv_blocks": res_kv,
            "resident_hs_blocks": res_hs,
            "resident_total_bytes": total_bytes,
            "zombie_joint_bytes": zombie_joint,
            "zombie_mixed_bytes": zombie_mixed,
            "zbr_joint": (zombie_joint / total_bytes) if total_bytes > 0 else 0.0,
            "zbr_mixed": (zombie_mixed / total_bytes) if total_bytes > 0 else 0.0,
            "host_hs_blocks": len(self._host_hs_cache) if self._host_hs_cache is not None else 0,
            "host_hs_used_bytes": self._host_hs_cache.used_bytes if self._host_hs_cache is not None else 0,
            "host_hs_hits": self._host_hs_cache.hits if self._host_hs_cache is not None else 0,
            "host_hs_misses": self._host_hs_cache.misses if self._host_hs_cache is not None else 0,
        }

    def shutdown(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        with self._lock:
            # Saves of normally finished requests may still wait for KV.
            for attempt in [*self._attempts.values(), *(save.attempt for save in self._saves.values())]:
                attempt.cancelled = True
        for executor in (self._read_executor, self._save_executor):
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
        super().shutdown()
        with self._lock:
            self._attempts.clear()
            self._delivered_until.clear()
            self._reads.clear()
            self._saves.clear()
            self._reserved_bytes = 0
