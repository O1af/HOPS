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

HOPS now uses a canonical preset-first schema with these top-level sections:

- `simulation`
- `pipeline`
- `hardware`
- `optimizer`
- `failure`
- `output`
- optional `overrides`

The default path is intentionally short: stages describe the work, `hardware` describes available devices and interconnect presets, and HOPS resolves the concrete capabilities internally.

```yaml
simulation:
  batches: 2
  microbatches: 4
  seed: 42

pipeline:
  schedule: 1f1b
  precision: bf16
  activation_mb: 50
  backward_factor: 2.0
  stages:
    - device: node0_gpu0
      weights_mb: 2048
      compute:
        mode: analytical
        tflop: 6.0
        memory_mb: 256.0
        efficiency:
          compute: 0.72
          memory: 0.85
        jitter: { type: normal, mean: 0.0, std: 0.15 }

    - device: node1_gpu0
      weights_mb: 3072
      compute:
        mode: explicit
        distribution: { type: normal, mean: 8.0, std: 0.3 }

hardware:
  devices:
    - { id: node0_gpu0, gpu: h100, node: node0, socket: 0 }
    - { id: node1_gpu0, gpu: a100, node: node1, socket: 0 }
  interconnect:
    same_node: nvlink
    cross_node: infiniband

optimizer:
  enabled: true
  gradient_mb: 100
  accumulation_steps: 1
  allreduce:
    algorithm: naive
  update: { type: normal, mean: 2.0, std: 0.3 }

failure:
  enabled: false

output:
  timeline: output/timeline.png
  dashboard: output/dashboard.png
  summary_json: output/summary.json
  trace_csv: output/trace.csv
```

### Compute modes

Each stage uses exactly one compute mode:

- `mode: explicit`
  Uses a measured or assumed latency distribution directly.
- `mode: analytical`
  Derives latency from stage workload plus device preset capability.

Analytical stages support optional remote memory placement:

```yaml
memory_placement:
  kind: socket
  node: node0
  socket: 1
```

or

```yaml
memory_placement:
  kind: device
  device: node0_gpu1
```

### Latency distributions

Stage distributions, optimizer update latency, and interconnect jitter support:

- `constant` — deterministic (`{ type: constant, value: 5.0 }`)
- `normal` — Gaussian, clamped to non-negative (`{ type: normal, mean: 5.0, std: 0.5 }`)
- `heavy_tailed` — Pareto, for modeling stragglers (`{ type: heavy_tailed, base: 5.0, alpha: 2.5 }`)
- `poisson` — discrete count (`{ type: poisson, lam: 5.0 }`)

### Hardware presets

Built-in device presets currently include:

- `h100`
- `a100`
- `l40s`
- `cpu-standard`

Built-in interconnect presets currently include:

- `nvlink`
- `pcie`
- `infiniband`
- `ethernet`

Use `overrides` only for advanced experiments:

```yaml
overrides:
  devices:
    - id: gpu0
      memory_mb: 40960
  links:
    - src: gpu0
      dst: gpu1
      bandwidth_gbps: 300
      latency_us: 7.0
      jitter: { type: constant, value: 0.0 }
```

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

Then set `pipeline.schedule: my_policy` in your YAML config.

## Metrics

HOPS reports:

- **Throughput** — micro-batches per ms and per s
- **End-to-end latency** — p50, p99, mean per micro-batch
- **Bubble ratio** — fraction of device-time spent idle
- **Per-stage / per-device / per-link utilization**
- **Communication overhead** — transfer time as a fraction of compute
- **Transfer contention** — peak concurrency and contended-transfer fraction
- **Optimizer step** — all-reduce time and weight update time (when enabled)
- **Failure impact** — count, total downtime, and lost-work slot
- **Peak memory per device**

Machine-readable output uses one canonical summary JSON schema and a separate raw event trace CSV.

## Visualization

When run without `--no-viz`, HOPS generates:

- `output/timeline.png` — Gantt chart showing per-device compute tasks (forward/backward), with failure markers
- `output/dashboard.png` — 8-panel summary including stage/device/link utilization, latency histogram, bubble ratio, compute vs communication, peak memory, and a key metrics panel

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
├── config.py            # Canonical config schema + parser/validation
├── core/
│   ├── event_engine.py    # Discrete event simulation loop (priority queue)
│   ├── pipeline.py        # Pipeline stages, micro-batches, forward/backward/optimizer dataflow
│   ├── scheduler.py       # Scheduling policies (GPipe, 1F1B) with plugin registry
│   ├── timing.py          # Timing and resource availability with failure-aware delays
│   └── types.py           # Core enums and dataclasses (EventKind, Phase, TaskStatus)
├── hardware/
│   ├── device.py          # GPU/CPU device abstractions
│   ├── network.py         # Communication links with bandwidth and jitter
│   └── topology.py        # Device graph with inferred locality-aware links
├── latency/
│   ├── distributions.py   # Configurable latency distributions (normal, heavy-tail, Pareto, Poisson)
│   └── compute_model.py   # Explicit and analytical stage latency modeling
├── failure/
│   └── engine.py          # Chaos Monkey-style failure injection with automatic recovery
├── metrics/
│   ├── collector.py       # Raw event recording
│   ├── analyzer.py        # Derived metrics and interval analysis
│   ├── summary.py         # Canonical summary dataclasses
│   ├── exporter.py        # Trace CSV serialization
│   └── reporter.py        # Text and JSON summary output
├── presets.py           # Built-in device and interconnect preset registry
├── runtime.py           # Runtime assembly from validated config
└── viz/
    ├── timeline.py        # Gantt-style timeline with failure markers
    └── dashboard.py       # 8-panel summary dashboard
```

| Directory | Purpose |
|-----------|---------|
| `configs/` | YAML experiment configurations |
| `tests/` | pytest test suite |
| `experiments/` | Experiment results |
| `output/` | Generated visualizations |

### Architecture

HOPS is a **discrete event simulator**. The core loop (`event_engine.py`) processes timestamped events from a priority queue. All modules plug into this loop:

1. **Config + Presets** resolve the short YAML schema into concrete hardware, links, and stage models
2. **Hardware** defines the physical topology — devices, inferred links, explicit link overrides, and locality
3. **Pipeline** orchestrates dataflow — forward passes, backward passes, activation transfers, and optimizer steps (all-reduce + weight update)
4. **Scheduler** decides task ordering per stage (pluggable via `register_scheduler`)
5. **Latency** provides stochastic computation and communication delays
6. **Failure** injects device and link faults with automatic recovery
7. **Metrics** records raw events, derives a canonical summary, and exports JSON/CSV outputs
8. **Viz** renders timeline and dashboard visualizations

Determinism is ensured by threading an explicit `np.random.Generator` (seeded from the YAML config) through all stochastic components.
