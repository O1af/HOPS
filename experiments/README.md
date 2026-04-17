# HOPS Validation Experiments

This directory contains small AWS ParallelCluster experiments for validating
HOPS against real Megatron-LM pipeline-parallel runs. The workflow is:

1. create or reuse a cluster config,
2. bootstrap `/home/ubuntu` once on the head node,
3. run a Slurm scenario,
4. compare HOPS output with Megatron logs or HOPS trace JSONL.

## Scenarios

- `05_h100_dual_node_pp2`
  `2 x p5.4xlarge`, one H100 per node, `PP=2`. This is the validated
  homogeneous H100 baseline.

- `06_h100_single_gpu_smoke`
  `1 x p5.4xlarge`, one H100, `PP=1`. This is a lightweight trace/conversion
  smoke test.

- `experiment_1/1_all_nodes`
  `2 x g5.2xlarge` A10G plus `2 x g6.4xlarge` L4, `PP=4`. This is the first
  G-family heterogeneous run and uses analytical HOPS estimates plus derived
  activation size.

## Cluster Configs

- `05_h100_dual_node_pp2/cluster_p5small_capacity_block_1f.yaml`
  Two-node P5 Capacity Block cluster.

- `06_h100_single_gpu_smoke/cluster_p5small_capacity_block_1gpu.yaml`
  Single-node P5 Capacity Block smoke cluster.

- `experiment_1/cluster_g5_g6_odcr_1f.yaml`
  Shared G-family On-Demand Capacity Reservation cluster for all
  `experiment_1/*` subsets.

Create the G-family cluster:

```bash
pcluster create-cluster \
  --cluster-name hops-experiment-1 \
  --cluster-configuration experiments/experiment_1/cluster_g5_g6_odcr_1f.yaml \
  --region us-east-1
```

Check the nodes:

```bash
sinfo
srun -p g-hetero --nodes=4 --ntasks=4 --ntasks-per-node=1 --gres=gpu:1 \
  bash -lc 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'
```

Expected for `experiment_1`: two A10G nodes and two L4 nodes.

## Fresh Head Node Bootstrap

Run this once on a blank head node. `/home/ubuntu` should be visible from the
compute nodes in these ParallelCluster configs.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
echo 'source "$HOME/.local/bin/env"' >> ~/.bashrc

cd /home/ubuntu
git clone https://github.com/O1af/HOPS.git
git clone https://github.com/O1af/Megatron-LM.git
```

Set up HOPS:

```bash
cd /home/ubuntu/HOPS
uv sync
uv run pytest tests -q
```

Set up Megatron:

```bash
cd /home/ubuntu
uv python install 3.12
uv venv --python 3.12 /home/ubuntu/megatron-env
source /home/ubuntu/megatron-env/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision torchaudio pyyaml pybind11 packaging ninja

cd /home/ubuntu/Megatron-LM
git switch main
git pull --ff-only origin main
grep -R "HOPS_TRACE" -n . | head -50
MAX_JOBS=4 uv pip install -e .
```

The `grep` should find HOPS trace hooks. If it does not, the Megatron checkout
will still train, but `megatron_trace/` will be empty and HOPS cannot import
per-stage real timings.

## Preflight

Check PyTorch and CUDA on the target partition:

```bash
srun -p g-hetero --nodes=4 --ntasks=4 --ntasks-per-node=1 --gres=gpu:1 \
  bash -lc 'source /home/ubuntu/megatron-env/bin/activate; hostname; python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"'
```

For a distributed NCCL smoke test, use a single Slurm-launched Python program
rather than an interactive `salloc` loop:

```bash
cat > /home/ubuntu/torch_dist_smoke_slurm.py <<'PY'
import os
import socket
import subprocess
import torch
import torch.distributed as dist

hosts = subprocess.check_output(
    ["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]],
    text=True,
).splitlines()

rank = int(os.environ["SLURM_PROCID"])
world = int(os.environ["SLURM_NTASKS"])
local_rank = int(os.environ["SLURM_LOCALID"])
os.environ["MASTER_ADDR"] = hosts[0]
os.environ.setdefault("MASTER_PORT", "29500")
os.environ["RANK"] = str(rank)
os.environ["WORLD_SIZE"] = str(world)
os.environ["LOCAL_RANK"] = str(local_rank)

torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
x = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(x)
print(
    f"host={socket.gethostname()} rank={rank}/{world} "
    f"gpu={torch.cuda.get_device_name(local_rank)} sum={x.item()}",
    flush=True,
)
dist.destroy_process_group()
PY
```

```bash
srun -p g-hetero --nodes=4 --ntasks=4 --ntasks-per-node=1 --gres=gpu:1 \
  bash -lc 'source /home/ubuntu/megatron-env/bin/activate; python /home/ubuntu/torch_dist_smoke_slurm.py'
```

Expected for the four-node G cluster: four ranks with `sum=10.0`.

## Run

Set the common environment:

```bash
cd /home/ubuntu/HOPS
export MEGATRON_DIR=/home/ubuntu/Megatron-LM
export MEGATRON_ENTRYPOINT=pretrain_gpt.py
export TRAIN_ENV_ACTIVATE=/home/ubuntu/megatron-env/bin/activate
```

Submit a scenario:

```bash
sbatch experiments/experiment_1/1_all_nodes/run.slurm
```

Monitor:

```bash
squeue -u "$USER"
sacct -j <job-id> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList
```

Cancel a bad run:

```bash
scancel <job-id>
```

## Inspect Outputs

Each run writes to `output/<job-id>/` under the scenario directory:

- `hops_summary.json`
- `hops_trace.csv`
- `hops_stdout.txt`
- `node_inventory.txt`
- `megatron_rank*.log` or scenario-specific Megatron logs
- `megatron_trace/*.jsonl` if HOPS Megatron trace hooks are active

Inspect the latest `experiment_1/1_all_nodes` run:

```bash
JOB_DIR=$(find experiments/experiment_1/1_all_nodes/output -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)
echo "$JOB_DIR"
ls -lh "$JOB_DIR"
ls -lh "$JOB_DIR/megatron_trace"
cat "$JOB_DIR/hops_stdout.txt"
```

Convert real Megatron traces when present:

```bash
uv run python -m hops.megatron_cli --job-dir "$JOB_DIR"
```

Copy results back to a local machine:

```bash
rsync -avz -e "ssh -i <key.pem>" \
  ubuntu@<head-node-host>:/home/ubuntu/HOPS/experiments/experiment_1/1_all_nodes/output/<job-id>/ \
  /Users/olaf/Documents/HOPS/experiments/experiment_1/1_all_nodes/output/<job-id>/
```

## Validation (three-variant comparison)

`experiment_1/1_all_nodes` follows a split between *committed intent* and
*generated calibration*:

- `hops.base.yaml` is the committed no-lookahead scenario spec. Never hand-edit
  calibrated numbers into it.
- `run.slurm` produces an immutable run artifact under
  `output/<run-job-id>/`.
- `link_bench.slurm` is submitted separately after `run.slurm` completes; its
  output lands under `output/<link-job-id>/calibration/link_bench/`.
- `experiments/tools/run_validation.py` post-processes those artifacts locally
  and emits `output/<run-job-id>/derived/{links.yaml,hops.link_calibrated.yaml,
  hops.trace_replay.yaml,hops_*_summary.json,comparison.json,report.md}`.

Three HOPS variants are always produced per run:

| variant | uses run data? | compute | links |
| --- | --- | --- | --- |
| `no_lookahead` | no | analytical (committed) | preset |
| `link_calibrated` | links only | analytical (committed) | measured |
| `trace_replay` | yes (posthoc) | per-stage forward fit from `megatron_trace` | measured |

`no_lookahead` and `link_calibrated` never see Megatron stage timings — that's
the overfit guardrail. `trace_replay` is a scheduler/topology mechanics sanity
check, not a predictive claim.

Run it locally after `rsync`ing both job dirs back:

```bash
uv run python experiments/tools/run_validation.py \
  --scenario experiments/experiment_1/1_all_nodes \
  --job-id <run-job-id> \
  --link-bench-dir experiments/experiment_1/1_all_nodes/output/<link-job-id>/calibration/link_bench
```

It prints `report.md` to stdout. If `--link-bench-dir` is omitted, the tool
looks under `output/<run-job-id>/calibration/link_bench/` first.

The individual tools can also be driven standalone for debugging — see
`experiments/tools/{parse_link_bench,materialize_hops_variant,derive_megatron_stats,compare_run}.py`.

## Notes

- G-family nodes in `experiment_1` use socket NCCL, not `aws-ofi-nccl`; the
  Slurm script sets the required NCCL environment.
- Empty `megatron_trace/` means Megatron ran without active HOPS trace hooks.
- `link_bench.py` is the communication calibration harness. Use it after the
  first successful run to replace guessed link values with measured bandwidth
  and latency.
