# HOPS

**Heterogeneous Optimized Pipeline Simulator** — a discrete-event simulator for
pipeline-parallel training on clusters with mixed GPUs and interconnects. HOPS
models compute, communication, scheduling, optimizer steps, transfer
contention, and failure recovery, and is validated against real Megatron-LM
traces.

Authors: Jimmy Dai, Joshua Hsueh, Olaf Dsouza, Rishith Seelam (University of
Michigan).

## Quick Start

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest tests/
uv run python main.py --no-viz
```

Run a specific configuration and emit machine-readable artifacts:

```bash
uv run python main.py \
  --config configs/default.yaml \
  --summary-json output/summary.json \
  --trace-csv output/trace.csv
```

Drop `--no-viz` to render the Gantt timeline and the eight-panel dashboard
(requires the dev extras installed by `uv sync`).

## Configuration

Simulations are described by YAML with a preset-first schema:

| Section      | Purpose                                                  |
| ------------ | -------------------------------------------------------- |
| `simulation` | batches, microbatches, seed                              |
| `pipeline`   | schedule, precision, activation size, per-stage compute  |
| `hardware`   | device and interconnect presets, topology                |
| `optimizer`  | optional all-reduce and update timing                    |
| `failure`    | optional device and link fault injection                 |
| `output`     | summary, trace, timeline, and dashboard paths            |
| `overrides`  | per-device or per-link overrides for sweeps              |

**Schedulers:** `gpipe`, `1f1b`, `zero_bubble`, and heterogeneity-aware
policies (`hops_hetero`, adaptive warmup).
**Device presets:** `h100`, `a100`, `a10g`, `l4`, `l40s`, `cpu-standard`.
**Interconnect presets:** `nvlink`, `pcie`, `infiniband`, `ethernet`.

A stage can use the **analytical** compute model (roofline over FLOPs, memory
traffic, launch overhead, and jitter) or the **explicit** model (configured
latency distribution). Minimal example:

```yaml
pipeline:
  schedule: 1f1b
  precision: bf16
  activation_mb: 50
  backward_factor: 2.0
  stages:
    - device: node0_gpu0
      weights_mb: 1792
      compute:
        mode: analytical
        tflop: 5.0
        memory_mb: 224.0
        efficiency: { compute: 0.76, memory: 0.88 }
```

See `configs/` for full examples, including single-node, DGX A100/H100, and
heterogeneous two-node setups.

## Outputs

- throughput and end-to-end latency
- bubble ratio and stage/device utilization
- per-link utilization, communication overhead, transfer contention
- optimizer timing (when enabled)
- failure count, downtime, recovery impact (when enabled)
- peak memory per device

Machine output: summary JSON plus raw trace CSV. Visual output: a Gantt-style
timeline and an eight-panel dashboard.

## Megatron Trace Workflow

Convert a Megatron job into HOPS-compatible artifacts:

```bash
uv run python -m hops.megatron_cli \
  --job-dir experiments/05_h100_dual_node_pp2/output/<job-id>
```

Validate managed fixtures on the **train** split:

```bash
uv run python experiments/tools/split_fixtures.py --apply
uv run python experiments/tools/validate_fixtures.py --split train
uv run python experiments/tools/inspect_fixture.py --fixture <name>
```

> The test split (`fixtures/test/`) is reserved for human-initiated checkpoint
> evaluation. Agents and model-tuning work must stay on the train split to
> keep the generalization measurement clean. See `AGENTS.md` and
> `AutoResearch.md` for the research loop.

## Repository Map

```text
src/hops/
  config.py             YAML schema and validation
  runtime.py            runtime assembly
  presets.py            device and interconnect catalogs
  core/                 event engine, pipeline orchestration, schedulers
  hardware/             devices, network, topology
  latency/              compute model and latency distributions
  metrics/              collection, analysis, reporting, export
  failure/              fault injection and recovery
  megatron/             Megatron trace import and comparison
  viz/                  timeline and dashboard rendering

configs/                runnable simulation examples
experiments/            AWS ParallelCluster scenarios and validation tooling
fixtures/               managed validation fixtures (train / test / diagnostic)
tests/                  pytest suite
AGENTS.md               AutoResearch playbook for agent contributors
AutoResearch.md         train-split modeling research log
```

## Development

```bash
uv run pytest tests/
```

For analytical-model changes, follow the train-split loop in `AutoResearch.md`:
observe fixture errors, make one physically grounded modeling change, validate
on train, update the golden only after metrics hold or improve, and log the
iteration.
