# HOPS — Heterogeneous Optimized Pipeline Simulator

A Python-based simulator for modeling pipeline parallel training across configurable hardware topologies, communication latencies, failure modes, and scheduling strategies.

## Authors

Jimmy Dai, Joshua Hsueh, Olaf Dsouza, Rishith Seelam — University of Michigan

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync
```

## Usage

```bash
uv run python main.py                        # run with default config
uv run python main.py --config my_config.yaml # run with custom config
uv run python main.py --no-viz                # skip visualization output
```

Run tests:

```bash
uv run pytest              # run all tests
uv run pytest tests/test_pipeline.py  # run a specific test file
uv run pytest -v           # verbose output
```

## Configuration

Simulations are defined via YAML configs in `configs/`. See `configs/default.yaml` for a full example. Key sections:

```yaml
simulation:
  num_microbatches: 8
  num_batches: 4
  seed: 42                # deterministic RNG seed

pipeline:
  stages:
    - id: 0
      device: node0_gpu0
      compute_latency: { type: normal, mean: 5.0, std: 0.5 }
  backward_factor: 2.0    # backward = 2x forward compute time

scheduler:
  policy: 1f1b             # "gpipe" or "1f1b", or any registered custom policy

hardware:
  devices:
    - id: node0_gpu0
      kind: gpu
      memory_mb: 81920
  links:
    - src: node0_gpu0
      dst: node0_gpu1
      bandwidth_gbps: 900
      base_latency_us: 1.0
      jitter: { type: normal, mean: 0.0, std: 0.1 }
  activation_size_mb: 50

optimizer:
  enabled: true
  gradient_size_mb: 100
  compute_latency: { type: normal, mean: 2.0, std: 0.3 }

failure:
  enabled: false
  check_interval: 10.0
  device_fail_prob: 0.001
  link_fail_prob: 0.0005
  recovery_time: 5.0
```

### Latency distributions

Stages and link jitter support configurable distributions:

- `constant` — deterministic (`{ type: constant, value: 5.0 }`)
- `normal` — Gaussian, clamped to non-negative (`{ type: normal, mean: 5.0, std: 0.5 }`)
- `heavy_tailed` — Pareto, for modeling stragglers (`{ type: heavy_tailed, base: 5.0, alpha: 2.5 }`)
- `poisson` — discrete count (`{ type: poisson, lam: 5.0 }`)

## Custom Scheduling Policies

HOPS ships with GPipe and 1F1B schedulers. To add your own:

```python
from hops import register_scheduler
from hops.core.scheduler import Scheduler, PipelineState
from hops.core.types import StageTask, Phase

class MyScheduler(Scheduler):
    def next_tasks(self, state: PipelineState) -> list[StageTask]:
        # state provides:
        #   state.is_task_ready(stage, mb, phase)
        #   state.completed_count(stage, phase)
        #   state.stage_is_busy(stage)
        #   state.stage_in_flight_count(stage)
        #   state.all_forwards_completed()
        ...

register_scheduler("my_policy", MyScheduler)
```

Then set `policy: my_policy` in your YAML config.

## Metrics

HOPS reports:

- **Throughput** — micro-batches per ms
- **End-to-end latency** — p50, p99, mean per micro-batch
- **Bubble ratio** — fraction of device-time spent idle
- **Per-stage utilization** — compute time / total time per stage
- **Communication overhead** — transfer time as a fraction of compute
- **Optimizer step** — all-reduce time and weight update time (when enabled)
- **Failure impact** — count and total downtime (when enabled)

## Visualization

When run without `--no-viz`, HOPS generates:

- `output/timeline.png` — Gantt chart showing per-device compute tasks (forward/backward), with failure markers
- `output/dashboard.png` — 4-panel summary: per-stage utilization, e2e latency histogram, bubble ratio, compute vs. communication breakdown

## Development

### Prerequisites

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install

```bash
git clone <repo-url> && cd HOPS
uv sync
```

This creates a `.venv`, installs all dependencies, and installs `hops` as an editable package.

### Add a dependency

```bash
uv add <package>              # runtime dependency
uv add --group dev <package>  # dev-only dependency
```

### Project Structure

```
src/hops/
├── core/
│   ├── event_engine.py    # Discrete event simulation loop (priority queue)
│   ├── pipeline.py        # Pipeline stages, micro-batches, forward/backward/optimizer dataflow
│   ├── scheduler.py       # Scheduling policies (GPipe, 1F1B) with plugin registry
│   ├── timing.py          # Timing and resource availability with failure-aware delays
│   └── types.py           # Core enums and dataclasses (EventKind, Phase, TaskStatus)
├── hardware/
│   ├── device.py          # GPU/CPU device abstractions
│   ├── network.py         # Communication links with bandwidth and jitter
│   └── topology.py        # Device graph with self-link optimization
├── latency/
│   ├── distributions.py   # Configurable latency distributions (normal, heavy-tail, Pareto, Poisson)
│   └── compute_model.py   # Per-stage computation time modeling with backward factor
├── failure/
│   └── engine.py          # Chaos Monkey-style failure injection with automatic recovery
├── metrics/
│   ├── collector.py       # Runtime statistics: compute, transfer, failure, in-flight records
│   └── reporter.py        # Summary report with throughput, bubbles, utilization, optimizer breakdown
└── viz/
    ├── timeline.py        # Gantt-style timeline with failure markers
    └── dashboard.py       # 4-panel summary dashboard
```

| Directory | Purpose |
|-----------|---------|
| `configs/` | YAML experiment configurations |
| `tests/` | pytest test suite (48 tests) |
| `experiments/` | Experiment results |
| `output/` | Generated visualizations |

### Architecture

HOPS is a **discrete event simulator**. The core loop (`event_engine.py`) processes timestamped events from a priority queue. All modules plug into this loop:

1. **Hardware** defines the physical topology — devices, links, and bandwidth
2. **Pipeline** orchestrates dataflow — forward passes, backward passes, activation transfers, and optimizer steps (all-reduce + weight update)
3. **Scheduler** decides task ordering per stage (pluggable via `register_scheduler`)
4. **Latency** provides stochastic computation and communication delays
5. **Failure** injects device and link faults with automatic recovery
6. **Metrics** collects records and computes derived metrics (throughput, bubble ratio, utilization)
7. **Viz** renders timeline and dashboard visualizations

Determinism is ensured by threading an explicit `np.random.Generator` (seeded from the YAML config) through all stochastic components.
