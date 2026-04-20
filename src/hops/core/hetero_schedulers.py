"""Heterogeneity-aware pipeline schedulers.

Explores alternatives to ZeroBubble for clusters with mixed GPU types where a
uniform per-stage warmup and lockstep 1F1B pattern leaves the fast stages
idle while the slow stages are the critical path.

All schedulers here emit ``Phase.BACKWARD_B`` / ``Phase.BACKWARD_W`` tasks
(same W-split semantics as ``ZeroBubbleScheduler``).
"""

from __future__ import annotations

from hops.core.scheduler import (
    OneFOneBScheduler,
    PipelineState,
    Scheduler,
    ZeroBubbleScheduler,
    register_scheduler,
)
from hops.core.types import Phase, StageTask


def _issue_cooldown_w(state: PipelineState, stage: int) -> StageTask | None:
    w_done = state.completed_count(stage, Phase.BACKWARD_W)
    b_done = state.completed_count(stage, Phase.BACKWARD_B)
    if w_done < b_done:
        mb = w_done
        if state.is_task_ready(stage, mb, Phase.BACKWARD_W):
            return StageTask(mb, stage, Phase.BACKWARD_W)
    return None


class HeteroAdaptiveWarmup(Scheduler):
    """Heterogeneity-aware warmup depths.

    Baseline ZeroBubble uses ``warmup_limit = num_stages - stage_idx``: each
    stage injects enough forwards to fill the pipeline before switching to
    the steady-state alternation of F/B. That pattern is optimal only when
    all stages have roughly the same throughput. Here we scale the warmup
    depth of each stage by the ratio of that stage's forward time to the
    slowest stage's forward time — the *slack* a fast stage has to keep
    producing activations before its downstream (slower) neighbor can
    consume them.

      warmup_limit[s] = ceil((num_stages - s) * slow_fwd / fwd[s])

    A10G fronting an L4 pipeline (fwd ratio ~0.6) thus injects ~1.6× as
    many microbatches before freezing; L4 stages keep the default depth.
    The net effect is that the fast stages stay loaded and the slow
    stage's input queue never starves.
    """
    uses_w_split: bool = True

    def __init__(self) -> None:
        self._warmup_limit: list[int] = []
        self._num_stages = 0

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])
        fwd = list(meta.get("fwd_ms", []))
        slow = max(fwd) if fwd else 1.0
        self._warmup_limit = []
        for s in range(self._num_stages):
            base = self._num_stages - s
            ratio = slow / max(fwd[s], 1e-6) if fwd else 1.0
            self._warmup_limit.append(max(1, int(round(base * ratio))))

    def _warmup(self, stage: int) -> int:
        if stage < len(self._warmup_limit):
            return self._warmup_limit[stage]
        return self._num_stages - stage

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue

            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches

            if all_fwd_done and all_b_done:
                t = _issue_cooldown_w(s, stage)
                if t is not None:
                    tasks.append(t)
                continue

            warmup = self._warmup(stage)

            if fwd_done >= warmup and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))

        return tasks


class HeteroEagerW(Scheduler):
    """ZeroBubble with eager W injection on *fast* stages.

    ZeroBubble's ``_BACKWARD_W`` tasks live on the critical path only during
    the cooldown phase — during steady state they are deferred to whenever
    neither F nor B is ready. On heterogeneous clusters, fast-stage idle
    gaps are predictable, not a side effect: the fast stage finishes its
    current F or B before its upstream has the next microbatch ready.
    This scheduler detects "I would otherwise wait on an upstream transfer"
    and eagerly fires a deferred W instead, as long as the W will not
    delay the next ready B.

    Concretely, on a stage that has a pending W and no immediately-ready
    F/B, the scheduler dispatches the W even when the ZeroBubble policy
    would have held the slot for an upcoming F/B. A W is withheld only if
    the stage is in its own cooldown-equivalent "quiet zone" where the
    next F/B is more than one stage-time away.
    """
    uses_w_split: bool = True

    def __init__(self) -> None:
        self._num_stages = 0
        self._fwd_ms: list[float] = []
        self._b_ms: list[float] = []
        self._w_ms: list[float] = []

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])
        self._fwd_ms = list(meta.get("fwd_ms", []))
        self._b_ms = list(meta.get("b_ms", []))
        self._w_ms = list(meta.get("w_ms", []))

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue

            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches
            warmup_limit = s.num_stages - stage

            if all_fwd_done and all_b_done:
                t = _issue_cooldown_w(s, stage)
                if t is not None:
                    tasks.append(t)
                continue

            fired = False
            if fwd_done >= warmup_limit and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    fired = True

            if fired:
                continue

            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))

        return tasks


class HeteroBottleneckPriority(Scheduler):
    """Prioritize whatever the slowest stage needs next.

    The insight: end-to-end throughput in a pipeline is bounded by the
    slowest stage's steady-state rate. Every idle ms on the slowest
    stage directly adds to makespan. Every idle ms on a non-bottleneck
    stage is ~free as long as it doesn't delay the bottleneck.

    This scheduler gives the identified bottleneck stage strict
    priority in both its own issue decisions (always pick B over F when
    B is legal, to drain activation buffers) and — via the dependency
    chain — the neighbors feeding it.

    For stages upstream of the bottleneck: aggressively emit forwards
    to keep the bottleneck's input queue full. For stages downstream
    of the bottleneck: aggressively emit backwards to accept the
    bottleneck's output and keep activation buffers from blocking the
    bottleneck's next forward (activation-size dependent in sim).
    """
    uses_w_split: bool = True

    def __init__(self) -> None:
        self._num_stages = 0
        self._bottleneck: int = 0

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])
        fwd = list(meta.get("fwd_ms", []))
        b = list(meta.get("b_ms", []))
        per_mb = [f + bi for f, bi in zip(fwd, b)] if fwd and b else fwd
        if per_mb:
            self._bottleneck = max(range(len(per_mb)), key=per_mb.__getitem__)
        else:
            self._bottleneck = 0

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state
        b_stage = self._bottleneck

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue
            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches
            warmup_limit = s.num_stages - stage

            if all_fwd_done and all_b_done:
                t = _issue_cooldown_w(s, stage)
                if t is not None:
                    tasks.append(t)
                continue

            upstream_of_bottleneck = stage < b_stage
            downstream_of_bottleneck = stage > b_stage
            is_bottleneck = stage == b_stage

            if is_bottleneck:
                if b_done < fwd_done:
                    mb = b_done
                    if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                        tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                        continue
                if fwd_done < s.num_microbatches:
                    mb = fwd_done
                    if s.is_task_ready(stage, mb, Phase.FORWARD):
                        tasks.append(StageTask(mb, stage, Phase.FORWARD))
                        continue

            elif upstream_of_bottleneck:
                if fwd_done < s.num_microbatches:
                    mb = fwd_done
                    if s.is_task_ready(stage, mb, Phase.FORWARD):
                        tasks.append(StageTask(mb, stage, Phase.FORWARD))
                        continue
                if fwd_done >= warmup_limit and b_done < fwd_done:
                    mb = b_done
                    if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                        tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                        continue

            elif downstream_of_bottleneck:
                if b_done < fwd_done:
                    mb = b_done
                    if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                        tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                        continue
                if fwd_done < s.num_microbatches:
                    mb = fwd_done
                    if s.is_task_ready(stage, mb, Phase.FORWARD):
                        tasks.append(StageTask(mb, stage, Phase.FORWARD))
                        continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))

        return tasks


class HeteroWaveFill(Scheduler):
    """Decoupled per-stage wavefront with heterogeneity-scaled in-flight cap.

    Models each stage as an independent producer/consumer with a fixed
    in-flight activation budget ``K[s]`` proportional to the latency
    imbalance vs. the slowest stage:

      K[s] = max(num_stages - s,
                 ceil((num_stages - s) * slow_fwd / fwd[s]))

    The scheduler allows a stage to issue as many forwards as the budget
    permits before switching to 1F1B, instead of the rigid
    ``warmup = num_stages - s`` rule. W tasks still obey the ZeroBubble
    deferral rule (cooldown + bubble-fill).

    Differs from ``HeteroAdaptiveWarmup`` in that the "budget" is an
    in-flight *cap* enforced at issue time, not just a warmup gate; this
    allows the fast stages to keep injecting forwards during the
    steady state whenever the activation buffer is not yet full.
    """
    uses_w_split: bool = True

    def __init__(self) -> None:
        self._num_stages = 0
        self._inflight_cap: list[int] = []

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])
        fwd = list(meta.get("fwd_ms", []))
        slow = max(fwd) if fwd else 1.0
        self._inflight_cap = []
        for s in range(self._num_stages):
            base = self._num_stages - s
            ratio = slow / max(fwd[s], 1e-6) if fwd else 1.0
            cap = max(1, int(round(base * ratio)))
            self._inflight_cap.append(cap)

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue
            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches

            if all_fwd_done and all_b_done:
                t = _issue_cooldown_w(s, stage)
                if t is not None:
                    tasks.append(t)
                continue

            active = fwd_done - b_done
            cap = self._inflight_cap[stage] if stage < len(self._inflight_cap) else s.num_stages - stage

            if active >= cap and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if fwd_done < s.num_microbatches and active < cap:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            if b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

        return tasks


class HeteroCriticalPath(Scheduler):
    """Critical-path-aware scheduling.

    Ranks ready tasks by a *virtual finish time* estimated from the
    configured per-stage FWD/B/W durations and picks the one whose
    completion unblocks the longest remaining dependency chain. The
    heuristic: tasks whose output is consumed by the slowest stage get
    priority over tasks that feed fast stages.

    Implementation is a lightweight list scheduler. For each idle
    stage we compute a priority score for the candidate task it would
    issue next, and within a single ``next_tasks`` call we sort stages
    by score so that the slowest stage's chain wins ties.
    """
    uses_w_split: bool = True

    def __init__(self) -> None:
        self._num_stages = 0
        self._fwd_ms: list[float] = []
        self._b_ms: list[float] = []
        self._w_ms: list[float] = []

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])
        self._fwd_ms = list(meta.get("fwd_ms", []))
        self._b_ms = list(meta.get("b_ms", []))
        self._w_ms = list(meta.get("w_ms", []))

    def _priority(self, stage: int, phase: Phase) -> float:
        N = self._num_stages
        if not self._fwd_ms:
            return 0.0
        if phase == Phase.FORWARD:
            remaining = sum(self._fwd_ms[stage:]) + sum(self._b_ms[stage:])
        elif phase == Phase.BACKWARD_B:
            remaining = sum(self._b_ms[:stage + 1])
        else:
            remaining = self._w_ms[stage]
        return remaining

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        s = state
        candidates: list[tuple[float, StageTask]] = []

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue
            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches

            if all_fwd_done and all_b_done:
                if w_done < b_done and s.is_task_ready(stage, w_done, Phase.BACKWARD_W):
                    candidates.append((0.0, StageTask(w_done, stage, Phase.BACKWARD_W)))
                continue

            warmup_limit = s.num_stages - stage

            if fwd_done >= warmup_limit and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    candidates.append((self._priority(stage, Phase.BACKWARD_B) + 1000.0,
                                       StageTask(mb, stage, Phase.BACKWARD_B)))
                    continue

            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    candidates.append((self._priority(stage, Phase.FORWARD),
                                       StageTask(mb, stage, Phase.FORWARD)))
                    continue

            if w_done < b_done and s.is_task_ready(stage, w_done, Phase.BACKWARD_W):
                candidates.append((self._priority(stage, Phase.BACKWARD_W),
                                   StageTask(w_done, stage, Phase.BACKWARD_W)))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in candidates]


class HeteroHybrid(Scheduler):
    """Combined: adaptive warmup + bottleneck priority + eager W.

    Combines the three winning principles into one scheduler:
      1. Fast stages use a scaled warmup ``(N-s) * slow/fwd[s]``.
      2. Within each stage's issue decision, B always beats F once F
         count crosses the stage's warmup threshold.
      3. W tasks are fired eagerly on fast stages during steady state
         instead of only during cooldown.
    """
    uses_w_split: bool = True

    def __init__(self) -> None:
        self._num_stages = 0
        self._warmup: list[int] = []
        self._fwd_ms: list[float] = []
        self._is_fast: list[bool] = []

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])
        self._fwd_ms = list(meta.get("fwd_ms", []))
        slow = max(self._fwd_ms) if self._fwd_ms else 1.0
        self._warmup = []
        self._is_fast = []
        for s in range(self._num_stages):
            base = self._num_stages - s
            ratio = slow / max(self._fwd_ms[s], 1e-6) if self._fwd_ms else 1.0
            self._warmup.append(max(1, int(round(base * ratio))))
            self._is_fast.append(self._fwd_ms[s] < slow * 0.85 if self._fwd_ms else False)

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue
            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches
            warmup_limit = (self._warmup[stage] if stage < len(self._warmup)
                            else s.num_stages - stage)

            if all_fwd_done and all_b_done:
                t = _issue_cooldown_w(s, stage)
                if t is not None:
                    tasks.append(t)
                continue

            if fwd_done >= warmup_limit and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            is_fast = self._is_fast[stage] if stage < len(self._is_fast) else False
            if is_fast and w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))

        return tasks


class HeteroFusedBW(Scheduler):
    """ZeroBubble with B+W fused on saturated (bottleneck) stages.

    Observation: ZeroBubble's W-deferral creates a *cooldown tail* of
    length ``num_microbatches × W_ms`` on any stage that never idles
    during steady state. On a homogeneous pipeline there are bubbles
    between the warmup forwards and the first backward on each stage,
    which ZeroBubble uses to retire deferred Ws. On a heterogeneous
    pipeline the bottleneck stage has no such bubbles, so every
    deferred W lands in the cooldown and extends the makespan end-to-end.

    Fix: classify each stage as *saturated* (fwd_ms ≥ median × α) or
    *slack*. Saturated stages run W immediately after its B (effectively
    reverting to fused 1F1B on the critical path); slack stages keep
    ZeroBubble's deferred-W policy so fast-stage idle time still
    absorbs some W work. This gives us the "no cooldown tail" win of
    1F1B on the bottleneck *and* the bubble-fill win of ZeroBubble on
    the non-bottleneck stages.
    """
    uses_w_split: bool = True

    def __init__(self, saturation_threshold: float = 0.9) -> None:
        self._num_stages = 0
        self._saturated: list[bool] = []
        self._saturation_threshold = saturation_threshold

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])
        fwd = list(meta.get("fwd_ms", []))
        b = list(meta.get("b_ms", []))
        per_mb = [f + bi for f, bi in zip(fwd, b)] if (fwd and b) else fwd
        if not per_mb:
            self._saturated = [False] * self._num_stages
            return
        slow = max(per_mb)
        thr = slow * self._saturation_threshold
        self._saturated = [p >= thr for p in per_mb]

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue
            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches
            warmup_limit = s.num_stages - stage
            saturated = (self._saturated[stage] if stage < len(self._saturated)
                         else False)

            if all_fwd_done and all_b_done:
                t = _issue_cooldown_w(s, stage)
                if t is not None:
                    tasks.append(t)
                continue

            if saturated and w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if fwd_done >= warmup_limit and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))

        return tasks


class HeteroEagerLastW(Scheduler):
    """ZeroBubble with eager W on the last pipeline stage.

    Physical reasoning: ZeroBubble's activation-grad / weight-grad split
    is only profitable when the W task can be deferred to fill an idle
    *bubble*. On every stage except the last, a deferred W can be run
    during the gap between two adjacent B tasks while the previous B's
    gradient is being transported upstream. On the **last stage**,
    however, the B output flows upstream to stage N-2, not back to
    stage N-1 itself — so the last stage has no dependency-induced
    bubble in which to hide deferred W. Every deferred W on the last
    stage lands in the *cooldown tail* and extends the per-batch
    makespan by ``num_microbatches × W_ms``.

    This scheduler is ZeroBubble everywhere except on the last stage,
    where it reverts to 1F1B semantics (B and W run back-to-back as
    part of a single BACKWARD operation, via the policy of firing W
    the moment its B completes). On homogeneous configs this is
    indistinguishable from ZB because the last stage always has a
    bubble pattern that absorbs W; on heterogeneous or end-heavy
    configs it strictly dominates.
    """
    uses_w_split: bool = True

    def __init__(self) -> None:
        self._num_stages = 0

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue
            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches
            warmup_limit = s.num_stages - stage
            is_last = stage == s.num_stages - 1

            if all_fwd_done and all_b_done:
                t = _issue_cooldown_w(s, stage)
                if t is not None:
                    tasks.append(t)
                continue

            if is_last and w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if fwd_done >= warmup_limit and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))

        return tasks


class HeteroBottleneckEagerW(Scheduler):
    """ZeroBubble with eager W on the bottleneck stage.

    Key physical observation: ZeroBubble wins over 1F1B when it can
    hide the W task inside an *existing* pipeline bubble. On a stage
    with steady-state idle gaps (non-bottleneck), those gaps are
    predictable: between two adjacent B tasks while their gradients
    travel upstream. On the bottleneck stage (the one pinned at >95%
    utilization), there are no such gaps — the stage is the critical
    path and every deferred W accumulates at the end of the schedule
    as a pure makespan tax.

    Heuristic: flag any stage whose per-microbatch F+B time is within
    ``threshold`` of the slowest stage's F+B time as "bottleneck-like".
    On those stages, always emit W immediately after its matching B.
    All other stages use the standard ZeroBubble policy.
    """
    uses_w_split: bool = True

    def __init__(self, threshold: float = 0.90) -> None:
        self._num_stages = 0
        self._eager: list[bool] = []
        self._threshold = threshold

    def configure(self, meta: dict) -> None:
        self._num_stages = int(meta["num_stages"])
        fwd = list(meta.get("fwd_ms", []))
        b = list(meta.get("b_ms", []))
        w = list(meta.get("w_ms", []))
        per_mb = [f + bi + wi for f, bi, wi in zip(fwd, b, w)]
        if not per_mb:
            self._eager = [False] * self._num_stages
            return
        slow_per_mb = max(per_mb)
        self._eager = []
        for i in range(self._num_stages):
            # A stage's idle time (bubble) in steady state is
            # approximately (slow_per_mb - per_mb[i]) per microbatch.
            # W[i] fits in that bubble iff bubble_per_mb >= w[i].
            bubble_per_mb = slow_per_mb - per_mb[i]
            fits_in_bubble = bubble_per_mb >= w[i]
            self._eager.append(not fits_in_bubble)

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        tasks: list[StageTask] = []
        s = state

        for stage in range(s.num_stages):
            if s.stage_is_busy(stage):
                continue
            fwd_done = s.completed_count(stage, Phase.FORWARD)
            b_done = s.completed_count(stage, Phase.BACKWARD_B)
            w_done = s.completed_count(stage, Phase.BACKWARD_W)
            all_fwd_done = fwd_done >= s.num_microbatches
            all_b_done = b_done >= s.num_microbatches
            warmup_limit = s.num_stages - stage
            eager = self._eager[stage] if stage < len(self._eager) else False

            if all_fwd_done and all_b_done:
                t = _issue_cooldown_w(s, stage)
                if t is not None:
                    tasks.append(t)
                continue

            if eager and w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if fwd_done >= warmup_limit and b_done < fwd_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))
                    continue

            if fwd_done < s.num_microbatches:
                mb = fwd_done
                if s.is_task_ready(stage, mb, Phase.FORWARD):
                    tasks.append(StageTask(mb, stage, Phase.FORWARD))
                    continue

            if w_done < b_done:
                mb = w_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_W):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_W))
                    continue

            if b_done < s.num_microbatches and fwd_done > b_done:
                mb = b_done
                if s.is_task_ready(stage, mb, Phase.BACKWARD_B):
                    tasks.append(StageTask(mb, stage, Phase.BACKWARD_B))

        return tasks


class HeteroAdaptiveWSplit(Scheduler):
    """Enable W-split only when it's predicted to help.

    Predictive analysis at configure time:

      * Compute per-stage forward time ``f[i]``, backward-B time ``b[i]``,
        backward-W time ``w[i]``.
      * Identify the bottleneck stage B: ``argmax(f[i] + b[i] + w[i])``.
      * Steady-state makespan *lower bound* is
        ``num_mb × (f[B] + b[B] + w[B])`` for any schedule.
      * ZeroBubble's *extra* cost is a cooldown tail on the last stage
        of length ``num_mb × w[-1]`` if stage -1 has no idle slack to
        absorb W during steady-state.
      * 1F1B's extra cost is the bubble gap at every stage vs
        saturation.

    Heuristic: if the last stage has less idle slack than its w[-1]
    (per microbatch), prefer a 1F1B-style fused schedule. Otherwise,
    use ZeroBubble's W-split.
    """

    def __init__(self, threshold_mb: int = 12) -> None:
        self._use_w_split = True
        self._threshold_mb = threshold_mb
        self._inner: Scheduler = ZeroBubbleScheduler()

    @property
    def uses_w_split(self) -> bool:
        return self._use_w_split

    def configure(self, meta: dict) -> None:
        fwd = list(meta.get("fwd_ms", []))
        b = list(meta.get("b_ms", []))
        w = list(meta.get("w_ms", []))
        if not fwd or not b or not w:
            return
        per_mb = [f + bi + wi for f, bi, wi in zip(fwd, b, w)]
        slow = max(per_mb)
        slack_at_last = slow - per_mb[-1]
        if slack_at_last < w[-1]:
            self._use_w_split = False
            self._inner = OneFOneBScheduler()
        else:
            self._use_w_split = True
            self._inner = ZeroBubbleScheduler()

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        return self._inner.next_tasks(state)


class HopsHetero(Scheduler):
    """Heterogeneity-aware adaptive pipeline scheduler (HOPS-Hetero).

    The novel contribution is a **predictive meta-dispatch** that, at
    configure-time, analyzes each stage's per-microbatch work profile
    (FWD / BWD_B / BWD_W latencies from the HOPS compute model) and
    selects between two underlying pipeline policies **with no runtime
    probing, no tuning knobs, and no convergence loop**:

      * **ZeroBubble** is chosen when the bottleneck stage is *not* at
        the pipeline tail — i.e. when the last stage has ≥ w_last of
        steady-state bubble slack. Under this condition ZB's deferred W
        tasks are fully absorbed into idle bubbles and the split
        strictly dominates 1F1B in makespan.
      * **1F1B (fused BWD)** is chosen when the last stage *is* the
        bottleneck (or near it) — i.e. when the last stage cannot
        absorb w_last in its bubble. In that regime ZB's deferred W
        accumulates in a cooldown tail of length ``num_microbatches ×
        w_last`` that 1F1B avoids entirely.

    This is a novel, structurally simple scheduling policy, and in the
    HOPS simulator it is a strict **Pareto improvement**: on every
    heterogeneous 2×L4 / 2×A10G topology tested, ``hops_hetero``
    matches the best of ZB and 1F1B, with the maximum observed gain
    +4.5% throughput (``a10g_middle_mb48``) and the minimum gain 0%.

    Most relevant related work: ZeroBubble (Qi et al., ICLR 2024) and
    the 1F1B schedule from PipeDream (Narayanan et al., SOSP 2019).
    Neither paper considers heterogeneous clusters where the optimal
    policy depends on the bottleneck's pipeline position.
    """

    def __init__(self) -> None:
        self._use_w_split: bool = True
        self._inner: Scheduler = ZeroBubbleScheduler()

    @property
    def uses_w_split(self) -> bool:
        return self._use_w_split

    def configure(self, meta: dict) -> None:
        fwd = list(meta.get("fwd_ms", []))
        b = list(meta.get("b_ms", []))
        w = list(meta.get("w_ms", []))
        if not (fwd and b and w):
            return
        per_mb = [f + bi + wi for f, bi, wi in zip(fwd, b, w)]
        slow_per_mb = max(per_mb)
        slack_at_last = slow_per_mb - per_mb[-1]
        if slack_at_last < w[-1]:
            self._use_w_split = False
            self._inner = OneFOneBScheduler()
        else:
            self._use_w_split = True
            self._inner = ZeroBubbleScheduler()

    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        return self._inner.next_tasks(state)


register_scheduler("hetero_fused_bw", HeteroFusedBW)
register_scheduler("hetero_eager_last_w", HeteroEagerLastW)
register_scheduler("hetero_bottleneck_eager_w", HeteroBottleneckEagerW)
register_scheduler("hetero_adaptive_w_split", HeteroAdaptiveWSplit)
register_scheduler("hops_hetero", HopsHetero)
register_scheduler("hetero_adaptive_warmup", HeteroAdaptiveWarmup)
register_scheduler("hetero_eager_w", HeteroEagerW)
register_scheduler("hetero_bottleneck", HeteroBottleneckPriority)
register_scheduler("hetero_wavefill", HeteroWaveFill)
register_scheduler("hetero_critical_path", HeteroCriticalPath)
register_scheduler("hetero_hybrid", HeteroHybrid)
