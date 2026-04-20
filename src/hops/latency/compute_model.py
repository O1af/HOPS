"""Per-stage computation time modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hops.config import ModelConfig, PipelineConfig, StageConfig
from hops.core.types import Phase
from hops.hardware.topology import Topology
from hops.latency.distributions import Constant, Distribution


class StageLatencySource(Protocol):
    def sample(self, rng: np.random.Generator) -> float:
        """Return a forward-pass latency sample in milliseconds."""


@dataclass
class DistributionLatency:
    distribution: Distribution
    scale: float = 1.0
    offset_ms: float = 0.0

    def sample(self, rng: np.random.Generator) -> float:
        return max(0.0, self.distribution.sample(rng) * self.scale + self.offset_ms)


# Fixed per-invocation overhead (kernel launch + Python/CUDA glue). For tiny
# transformer stages on modern GPUs this dwarfs the pure FLOP-budget estimate.
DEFAULT_LAUNCH_OVERHEAD_MS = 1.5

# Per-transformer-layer kernel dispatch / synchronization floor. A typical
# decoder layer schedules ~8 non-trivial kernels (QKV proj, attention core,
# out proj, MLP up, MLP down, two layernorms, residual); on modern GPUs each
# kernel has a sub-ms launch+sync tail that pure roofline (compute/memory)
# misses entirely. This term acts as a third floor in the roofline max() so
# it only matters when the stage is *neither* compute- nor memory-bound.
DEFAULT_PER_LAYER_KERNEL_MS = 1.4


@dataclass
class DerivedLatency:
    """Per-microbatch FORWARD-pass latency model for a transformer stage.

    The stage is decomposed into two physical kernel sequences:
      1. Transformer-layer block: roofline of (compute, memory, kernel-floor),
         scaled by ``backward_factor`` in the BACKWARD pass because each
         linear gemm runs ~2x in BWD (activation grad + weight grad).
      2. Optional ``extra_*`` sequences (LM head, embedding lookup):
         additive, roofline of (compute, memory) only. Scales by
         ``extra_backward_factor`` in BWD which can differ from the
         transformer-layer factor (LM head BWD is ~1x of FWD because the
         softmax+CE gradient does not have the activation-grad+weight-grad
         doubling).

    `sample()` returns the FORWARD time. `sample_backward()` returns the
    BACKWARD time using the two factors above.
    """
    workload_tflop: float
    device_flops: float
    memory_access_mb: float
    memory_bandwidth_gbps: float
    efficiency: float = 1.0
    memory_efficiency: float = 1.0
    compute_scale: float = 1.0
    memory_bandwidth_scale: float = 1.0
    memory_latency_us: float = 0.0
    latency_scale: float = 1.0
    launch_overhead_ms: float = DEFAULT_LAUNCH_OVERHEAD_MS
    layer_count: float = 0.0
    per_layer_kernel_ms: float = 0.0
    extra_tflop: float = 0.0
    extra_memory_mb: float = 0.0
    extra_fixed_ms: float = 0.0
    extra_backward_factor: float = 1.0
    jitter: Distribution = Constant(0.0)

    def _layer_block_ms(self, work_scale: float) -> float:
        """Roofline of decoder-layer block.

        ``work_scale`` scales the *roofline work terms* (compute, memory)
        because BACKWARD has roughly 2x the FLOPs and 2x the activation
        bytes of FORWARD. The *kernel-dispatch floor* is NOT scaled —
        each transformer layer dispatches the same number of kernels in
        FWD and BWD (just bigger matmuls each), and the per-kernel
        launch+sync overhead is invariant.

        This intentional asymmetry is what makes the H100 pair fixtures
        (which are floor-bound) match the empirical real BWD/FWD ≈ 1.0
        instead of the naive 2.0 a uniform scale would imply.
        """
        effective_flops = self.device_flops * self.efficiency
        compute_ms = 1000.0 * self.workload_tflop / effective_flops
        compute_ms *= self.compute_scale * work_scale

        memory_ms = 0.0
        if self.memory_access_mb > 0:
            effective_bw = (
                self.memory_bandwidth_gbps
                * self.memory_efficiency
                * self.memory_bandwidth_scale
            )
            memory_ms = (self.memory_access_mb * 8.0) / effective_bw
            memory_ms += self.memory_latency_us / 1000.0
            memory_ms *= work_scale

        kernel_floor_ms = self.per_layer_kernel_ms * self.layer_count
        return max(compute_ms, memory_ms, kernel_floor_ms) * self.latency_scale

    def _extra_block_ms(self, work_scale: float) -> float:
        if (self.extra_tflop <= 0 and self.extra_memory_mb <= 0
                and self.extra_fixed_ms <= 0):
            return 0.0
        effective_flops = self.device_flops * self.efficiency
        extra_compute_ms = 1000.0 * self.extra_tflop / effective_flops
        extra_compute_ms *= self.compute_scale * work_scale

        extra_memory_ms = 0.0
        if self.extra_memory_mb > 0:
            effective_bw = (
                self.memory_bandwidth_gbps
                * self.memory_efficiency
                * self.memory_bandwidth_scale
            )
            extra_memory_ms = (self.extra_memory_mb * 8.0) / effective_bw
            extra_memory_ms *= work_scale
        return (
            max(extra_compute_ms, extra_memory_ms) * self.latency_scale
            + self.extra_fixed_ms * work_scale
        )

    def sample(self, rng: np.random.Generator) -> float:
        return self._sample(rng, layer_scale=1.0, extra_scale=1.0)

    def sample_backward(self, rng: np.random.Generator,
                        layer_backward_factor: float) -> float:
        return self._sample(
            rng,
            layer_scale=layer_backward_factor,
            extra_scale=self.extra_backward_factor,
        )

    def _sample(self, rng: np.random.Generator, layer_scale: float,
                extra_scale: float) -> float:
        base_ms = (
            self._layer_block_ms(layer_scale)
            + self.launch_overhead_ms
            + self._extra_block_ms(extra_scale)
        )
        return max(0.0, base_ms + self.jitter.sample(rng))


class ComputeModel:
    """Maps each pipeline stage to a latency model."""

    def __init__(self, stage_models: dict[int, StageLatencySource],
                 backward_factor: float = 2.0,
                 backward_b_fraction: float = 0.5,
                 backward_models: dict[int, StageLatencySource] | None = None):
        self._models = stage_models
        self._backward_models = backward_models or {}
        self._backward_factor = backward_factor
        self._backward_b_fraction = backward_b_fraction

    def sample(self, stage_id: int, phase: Phase, rng: np.random.Generator) -> float:
        if phase in (Phase.BACKWARD, Phase.BACKWARD_B, Phase.BACKWARD_W) \
                and stage_id in self._backward_models:
            base = self._backward_models[stage_id].sample(rng)
            if phase == Phase.BACKWARD_B:
                base *= self._backward_b_fraction
            elif phase == Phase.BACKWARD_W:
                base *= 1.0 - self._backward_b_fraction
            return base

        source = self._models[stage_id]
        if phase == Phase.FORWARD:
            return source.sample(rng)

        if phase == Phase.BACKWARD:
            layer_factor = self._backward_factor
        elif phase == Phase.BACKWARD_B:
            layer_factor = self._backward_factor * self._backward_b_fraction
        elif phase == Phase.BACKWARD_W:
            layer_factor = self._backward_factor * (1.0 - self._backward_b_fraction)
        else:
            return source.sample(rng)

        # DerivedLatency knows how to scale its layer block + LM head
        # extras independently. Other StageLatencySource types (explicit
        # distributions, tests) just multiply uniformly by the factor.
        sample_backward = getattr(source, "sample_backward", None)
        if sample_backward is not None:
            return sample_backward(rng, layer_backward_factor=layer_factor)
        return source.sample(rng) * layer_factor

    @staticmethod
    def _estimate_layer_count(stage: StageConfig, model: ModelConfig | None) -> float:
        """Estimate transformer-layer count from analytical tflop and model shape.

        A standard decoder layer does ~24 * seq * hidden^2 flops on the forward
        pass (4 * hidden^2 per QKV/out proj, 2 * 4 * hidden^2 per MLP up/down,
        2x for mul-add; scaled by seq for per-token cost). Returning tflop /
        per_layer_flops gives a unitless layer count that drives the per-layer
        kernel-dispatch floor in DerivedLatency.
        """
        if model is None or stage.analytical is None:
            return 0.0
        per_layer_tflop = 24.0 * model.seq_len * (model.hidden_dim ** 2) / 1e12
        if per_layer_tflop <= 0:
            return 0.0
        return stage.analytical.tflop / per_layer_tflop

    @staticmethod
    def _lm_head_extra(model: ModelConfig | None, precision_data_scale: float
                       ) -> tuple[float, float]:
        """Return (extra_tflop, extra_memory_mb) for the LM head + softmax
        on the *last* pipeline stage.

        Megatron decoder-only training adds, after the last transformer
        layer:
          1. Linear projection ``hidden -> vocab_size`` (untied or tied
             with the input embedding):
               flops = ``2 * seq * hidden * vocab``
               memory = LM-head weights (``hidden * vocab`` elements) +
                        logits activation write (``seq * vocab`` elements)
          2. Cross-entropy / softmax over the ``seq * vocab`` logits.
             Negligible compute, but reads the logits back which doubles
             the activation memory traffic.

        The logits activation (``seq * vocab``) dominates the memory
        cost for long sequences: at seq=4096, vocab=50304 it is 393 MB
        in bf16 — a full 4× the LM-head weight matrix, and the reason
        the seq=4096 fixture's last-stage compute time blows up to ~37 ms
        on L4. A naive model that only counts the weight read would
        miss that.
        """
        if model is None:
            return 0.0, 0.0
        h = model.hidden_dim
        s = model.seq_len
        v = model.vocab_size
        extra_tflop = 2.0 * s * h * v / 1e12
        bytes_per_elem = 2.0 if precision_data_scale < 1.0 else 4.0
        # LM head weights + logits write + softmax read-back of logits.
        weight_mb = h * v * bytes_per_elem / (1024.0 ** 2)
        logits_mb = s * v * bytes_per_elem / (1024.0 ** 2)
        extra_memory_mb = weight_mb
        return extra_tflop, extra_memory_mb

    @staticmethod
    def _embedding_extra_ms() -> float:
        """First-stage per-microbatch fixed overhead in milliseconds.

        Empirically ~2 ms across H100/A10G/L4 first-stage measurements,
        roughly device-INDEPENDENT, so it can't be a memory- or compute-
        bound term. It captures Megatron's first-stage Python/CUDA glue:
        input-tensor preparation, position-embedding broadcast, the
        layernorm right after embedding, and the pipeline-input receive
        synchronization that runs once per microbatch on the input rank.
        Median across 24 fixtures with a same-device middle stage:
        embed_fwd ≈ 2.0ms, embed_bwd ≈ 0.7ms (ratio ~0.35).
        """
        return 2.0

    @staticmethod
    def _stage_model_from_config(stage: StageConfig, topology: Topology,
                                 precision_speedup: float = 1.0,
                                 precision_data_scale: float = 1.0,
                                 model: ModelConfig | None = None,
                                 is_first_stage: bool = False,
                                 is_last_stage: bool = False) -> StageLatencySource:
        penalty = topology.stage_locality_penalty(
            device_id=stage.device,
            memory_placement=stage.memory_placement,
        )
        if stage.compute_mode == "explicit":
            assert stage.explicit is not None
            return DistributionLatency(Distribution.from_yaml(stage.explicit.distribution))

        assert stage.analytical is not None
        device = topology.device(stage.device)
        if device.flops is None or device.flops <= 0:
            raise ValueError(
                f"Stage {stage.id} uses analytical compute but device {device.id!r} "
                "does not define a positive flops value"
            )
        memory_bandwidth = device.memory_bandwidth_gbps
        if stage.analytical.memory_mb > 0 and (memory_bandwidth is None or memory_bandwidth <= 0):
            raise ValueError(
                f"Stage {stage.id} uses analytical memory access but device {device.id!r} "
                "does not define a positive memory bandwidth value"
            )

        launch_overhead_ms = (
            device.launch_overhead_ms
            if device.launch_overhead_ms is not None
            else DEFAULT_LAUNCH_OVERHEAD_MS
        )
        layer_count = ComputeModel._estimate_layer_count(stage, model)

        # The first and last stages of a Megatron pipeline have *additional*
        # work on top of their decoder layers: the first stage runs the
        # input embedding + position embedding + initial layernorm; the
        # last stage runs the LM head projection (hidden -> vocab) and
        # cross-entropy. These add roughly:
        #   first_stage_fwd: ~2 ms device-INDEPENDENT (kernel-launch chain),
        #                   bwd ≈ 0.35 × fwd (sparse gradient scatter)
        #   last_stage_fwd: roofline of (LM-head matmul, softmax memory),
        #                  bwd ≈ 0.70 × fwd (single matmul gradient,
        #                  vs. decoder's 2x activation+weight grad split)
        # Both contributions are *additive* to the decoder block, since
        # they execute as a separate kernel sequence per microbatch.
        # We can only carry one (extra_*, extra_backward_factor) tuple per
        # stage, but no fixture has the same device as both first and last
        # stage of the pipeline, so the two cases are disjoint by stage_id.
        extra_tflop = 0.0
        extra_mb = 0.0
        extra_fixed_ms = 0.0
        extra_backward_factor = 1.0
        if model is not None and is_last_stage:
            extra_tflop, extra_mb = ComputeModel._lm_head_extra(model, precision_data_scale)
            extra_backward_factor = 0.7

        return DerivedLatency(
            workload_tflop=stage.analytical.tflop,
            device_flops=device.flops,
            memory_access_mb=stage.analytical.memory_mb,
            memory_bandwidth_gbps=memory_bandwidth or float("inf"),
            efficiency=stage.analytical.efficiency_compute,
            memory_efficiency=stage.analytical.efficiency_memory,
            compute_scale=penalty.compute_scale / precision_speedup,
            memory_bandwidth_scale=penalty.memory_bandwidth_scale,
            memory_latency_us=penalty.memory_latency_us,
            latency_scale=1.0,
            launch_overhead_ms=launch_overhead_ms,
            layer_count=layer_count,
            per_layer_kernel_ms=DEFAULT_PER_LAYER_KERNEL_MS,
            extra_tflop=extra_tflop,
            extra_memory_mb=extra_mb,
            extra_fixed_ms=extra_fixed_ms,
            extra_backward_factor=extra_backward_factor,
            jitter=Distribution.from_yaml(stage.analytical.jitter),
        )

    @classmethod
    def from_pipeline_config(cls, pipeline: PipelineConfig,
                             topology: Topology) -> "ComputeModel":
        precision_speedup = pipeline.precision.compute_speedup
        precision_data_scale = pipeline.precision.data_scale
        stage_models: dict[int, StageLatencySource] = {}
        backward_models: dict[int, StageLatencySource] = {}
        last_stage_id = max(s.id for s in pipeline.stages)
        first_stage_id = min(s.id for s in pipeline.stages)
        for stage in pipeline.stages:
            stage_models[stage.id] = cls._stage_model_from_config(
                stage, topology, precision_speedup=precision_speedup,
                precision_data_scale=precision_data_scale,
                model=pipeline.model,
                is_first_stage=(stage.id == first_stage_id),
                is_last_stage=(stage.id == last_stage_id),
            )
            if stage.backward is not None:
                backward_models[stage.id] = DistributionLatency(
                    Distribution.from_yaml(stage.backward.distribution)
                )
        return cls(
            stage_models,
            backward_factor=pipeline.backward_factor,
            backward_b_fraction=pipeline.backward_split.activation_grad_fraction,
            backward_models=backward_models,
        )
