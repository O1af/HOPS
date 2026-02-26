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
uv run python main.py
```

Run tests:

```bash
uv run pytest
```

## Development

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) (handles Python, virtualenvs, and dependencies):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install

Clone the repo and sync all dependencies (including dev tools like pytest and matplotlib):

```bash
git clone <repo-url> && cd HOPS
uv sync
```

This creates a `.venv`, installs all dependencies, and installs `hops` as an editable package. No need to `pip install` anything — `uv sync` is the single command.

### Run

```bash
uv run python main.py
```

Use `uv run` to execute any command inside the project's virtualenv without activating it manually.

### Test

```bash
uv run pytest              # run all tests
uv run pytest tests/test_pipeline.py  # run a specific test file
uv run pytest -v           # verbose output
```

Tests live in `tests/` and mirror the `src/hops/` module structure.

### Add a dependency

```bash
uv add <package>           # runtime dependency
uv add --group dev <package>  # dev-only dependency
```

Both update `pyproject.toml` and `uv.lock` automatically.

### Project Structure

```
src/hops/
├── core/
│   ├── event_engine.py    # Discrete event simulation loop (priority queue by timestamp)
│   ├── pipeline.py        # Pipeline stages, micro-batches, forward/backward dataflow
│   └── scheduler.py       # Scheduling policies (1F1B, ZeroBubble, etc.)
├── hardware/
│   ├── device.py          # GPU/CPU device abstractions (compute caps, memory)
│   ├── topology.py        # Node/device graph, NUMA modeling
│   └── network.py         # Inter/intra-node communication links and bandwidths
├── latency/
│   ├── distributions.py   # Configurable latency distributions (normal, heavy-tail, etc.)
│   └── compute_model.py   # Per-stage computation time modeling
├── failure/
│   ├── engine.py          # Chaos Monkey-style failure injection during simulation
│   └── recovery.py        # Recovery strategies (re-scheduling, backpressure)
├── metrics/
│   ├── collector.py       # Runtime stats accumulation (per-stage, per-device)
│   └── reporter.py        # Throughput, bubble ratio, utilization calculations
└── viz/
    ├── timeline.py        # Gantt-style pipeline timeline plots
    └── dashboard.py       # Summary visualization dashboard
```

| Directory | Purpose |
|-----------|---------|
| `configs/` | YAML experiment configurations (topology, schedule, latency params) |
| `tests/` | pytest test suite |
| `experiments/` | Experiment runner scripts and results |

### Architecture Overview

HOPS is a **discrete event simulator**. The core loop (`event_engine.py`) processes timestamped events (compute completion, message arrival, failure injection) from a priority queue. All other modules plug into this loop:

1. **Hardware** defines the physical topology — devices, nodes, and communication links
2. **Pipeline** models the logical dataflow — stages and micro-batches moving through the topology
3. **Scheduler** decides when to issue micro-batches and assign stages to devices
4. **Latency** provides stochastic computation/communication delays
5. **Failure** injects faults and triggers recovery strategies
6. **Metrics** observes everything and produces performance reports
7. **Viz** renders metrics into plots and dashboards

Experiments are defined declaratively via YAML configs in `configs/`.
