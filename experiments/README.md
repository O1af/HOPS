# Heterogeneous Cluster Validation Experiments

These assets are designed for a small ParallelCluster validation campaign where:

- one compute node is a `p5.48xlarge` (H100)
- one compute node is a `p4d.24xlarge` (A100)
- you want the real Megatron execution topology to match the HOPS topology as closely as possible

## Scenarios

1. `05_h100_dual_node_pp2`
   Two-node H100 baseline using 1 GPU per node and `PP=2`.
   Purpose: validate HOPS against a capacity-constrained H100 setup such as `2 x p5.4xlarge`,
   where each stage runs on a different node and all pipeline communication is cross-node.

2. `06_h100_single_gpu_smoke`
   Single-node H100 smoke test using 1 GPU and `PP=1`.
   Purpose: validate Megatron tracing, raw JSONL emission, and HOPS conversion plumbing
   without requiring a second GPU or any cross-stage communication.

## Files

- `link_bench.py`
  NCCL-based point-to-point and all-reduce microbenchmark. Use this to fit HOPS `overrides.links`.
- `*/hops.yaml`
  Initial HOPS template for the scenario. The explicit stage means are placeholders that should be updated after the first calibration run.
- `*/run.slurm`
  Slurm batch script intended for submission from the ParallelCluster head node.
- `05_h100_dual_node_pp2/cluster_p5small_capacity_block_1f.yaml`
  Dedicated ParallelCluster config for the two-node H100 baseline.
- `06_h100_single_gpu_smoke/cluster_p5small_capacity_block_1gpu.yaml`
  Dedicated ParallelCluster config for the single-node H100 smoke test.

## Fresh Cluster Bootstrap

On a fresh head node, do not assume `/home/ubuntu/megatron-env` already exists.
Bootstrap the Python tooling with `uv`, then create the Megatron training environment
explicitly:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

cd /home/ubuntu
git clone https://github.com/O1af/HOPS.git
git clone https://github.com/O1af/Megatron-LM.git

cd /home/ubuntu/HOPS
uv sync

cd /home/ubuntu
uv python install 3.12
uv venv --python 3.12 /home/ubuntu/megatron-env
source /home/ubuntu/megatron-env/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision torchaudio pyyaml pybind11

cd /home/ubuntu/Megatron-LM
git checkout hops-tracing
MAX_JOBS=4 uv pip install -e .
```

## Prerequisites

- This repo is visible on the compute nodes through shared storage.
- `uv` is installed on the head node and available on compute nodes via `source "$HOME/.local/bin/env"`.
- `MEGATRON_DIR` points to a shared Megatron-LM checkout.
- `TRAIN_ENV_ACTIVATE` points to the training environment activation script, typically `/home/ubuntu/megatron-env/bin/activate`.
- The training environment referenced by `TRAIN_ENV_ACTIVATE` must provide PyTorch and
  a working `python -m torch.distributed.run`.
- The nodes have a working CUDA/NCCL stack.
- The head node can submit Slurm jobs.
- For the heterogeneous scripts, the allocation contains exactly one H100 node and one A100 node.

## Preflight Check

Before submitting any Megatron-backed experiment, verify the training environment on the
head node:

```bash
source "$HOME/.local/bin/env"
source /home/ubuntu/megatron-env/bin/activate
which python
python -c "print('python ok')"
timeout 30s python -c "import torch; print(torch.__version__)"
timeout 30s python -c "import torch.distributed.run; print('torch.distributed.run import works')"
```

If either timed import fails or times out, fix the training environment before submitting.
Otherwise the Slurm job will fail on the compute nodes when it tries to launch Megatron.

## Submission Examples

Two-node H100 baseline:

```bash
export MEGATRON_DIR=/home/ubuntu/Megatron-LM
export TRAIN_ENV_ACTIVATE=/home/ubuntu/megatron-env/bin/activate
sbatch experiments/05_h100_dual_node_pp2/run.slurm
```

Single-node H100 smoke test:

```bash
export MEGATRON_DIR=/home/ubuntu/Megatron-LM
export TRAIN_ENV_ACTIVATE=/home/ubuntu/megatron-env/bin/activate
sbatch experiments/06_h100_single_gpu_smoke/run.slurm
```

Single-node H100 smoke-test cluster creation:

```bash
pcluster create-cluster \
  --cluster-name <cluster-name> \
  --cluster-configuration experiments/06_h100_single_gpu_smoke/cluster_p5small_capacity_block_1gpu.yaml \
  --region us-east-1
```

If your cluster does not expose both node types from one partition, keep the inner distributed-launch logic from the heterogeneous scripts but translate the allocation to your site-specific Slurm constraints.

## Reproducing Run 26

This is the exact workflow used to get the first successful real Megatron run for
`05_h100_dual_node_pp2` on AWS ParallelCluster with `2 x p5.4xlarge` H100 nodes.
Use it as a runbook. Some AWS IDs and Capacity Block offering IDs are time-specific
and will need to be re-discovered, but the sequence is the important part.

### 1. Find and purchase a 2-node P5 Capacity Block

Look for a `p5.4xlarge` offering for `instance-count=2` in `us-east-1`:

```bash
aws ec2 describe-capacity-block-offerings \
  --region us-east-1 \
  --instance-type p5.4xlarge \
  --capacity-duration-hours 24 \
  --instance-count 2 \
  --query 'CapacityBlockOfferings[*].[CapacityBlockOfferingId,AvailabilityZone,StartDate,EndDate,CapacityBlockDurationHours,CapacityBlockDurationMinutes,UpfrontFee]' \
  --output table
```

Purchase the chosen offering:

```bash
aws ec2 purchase-capacity-block \
  --region us-east-1 \
  --capacity-block-offering-id <capacity-block-offering-id> \
  --instance-platform Linux/UNIX
```

Record the returned `CapacityReservationId`, then wait for it to become active:

```bash
aws ec2 describe-capacity-reservations \
  --capacity-reservation-ids <capacity-reservation-id> \
  --region us-east-1 \
  --query 'CapacityReservations[*].[CapacityReservationId,State,AvailabilityZone,StartDate,EndDate,TotalInstanceCount,AvailableInstanceCount]' \
  --output table
```

For the successful run described here, the reservation was purchased in `<availability-zone>`
and later attached to `cluster_p5small_capacity_block_1f.yaml`.

### 2. Prepare `us-east-1f` networking

You need:

- a VPC ID such as `<vpc-id>`
- a public subnet in the target AZ such as `<public-subnet-id>`
- a private compute subnet in the same AZ such as `<private-subnet-id>`

Create the private subnet:

```bash
aws ec2 create-subnet \
  --vpc-id <vpc-id> \
  --availability-zone <availability-zone> \
  --cidr-block <private-subnet-cidr>
```

Allocate an Elastic IP and create a NAT gateway in the public `1f` subnet:

```bash
aws ec2 allocate-address --domain vpc
```

```bash
aws ec2 create-nat-gateway \
  --subnet-id <public-subnet-id> \
  --allocation-id <allocation-id>
```

Wait for the NAT to become available:

```bash
aws ec2 describe-nat-gateways \
  --filter Name=vpc-id,Values=<vpc-id> \
  --query 'NatGateways[*].[NatGatewayId,State,SubnetId]' \
  --output table
```

Create and associate a private route table:

```bash
aws ec2 create-route-table \
  --vpc-id <vpc-id>
```

```bash
aws ec2 create-route \
  --route-table-id <route-table-id> \
  --destination-cidr-block 0.0.0.0/0 \
  --nat-gateway-id <nat-gateway-id>
```

```bash
aws ec2 associate-route-table \
  --route-table-id <route-table-id> \
  --subnet-id <private-subnet-id>
```

### 3. Create the ParallelCluster cluster

Use the checked-in config:

- `experiments/05_h100_dual_node_pp2/cluster_p5small_capacity_block_1f.yaml`

Dry-run first:

```bash
pcluster create-cluster \
  --cluster-name <cluster-name> \
  --cluster-configuration experiments/05_h100_dual_node_pp2/cluster_p5small_capacity_block_1f.yaml \
  --region us-east-1 \
  --dryrun true
```

Then create the cluster:

```bash
pcluster create-cluster \
  --cluster-name <cluster-name> \
  --cluster-configuration experiments/05_h100_dual_node_pp2/cluster_p5small_capacity_block_1f.yaml \
  --region us-east-1
```

Monitor creation:

```bash
pcluster list-clusters --region us-east-1
```

```bash
pcluster get-cluster-stack-events \
  --cluster-name <cluster-name> \
  --region us-east-1
```

### 4. SSH to the head node and verify the H100 nodes

SSH to the head node using your key and hostname:

```bash
ssh -i "<path-to-private-key>" ubuntu@<head-node-hostname>
```

Verify Slurm sees the nodes:

```bash
sinfo
sinfo -R
```

Verify H100 availability:

```bash
srun -p fast-gpu --nodes=1 --ntasks=1 bash -lc 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1'
```

```bash
srun -p fast-gpu --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1'
```

### 5. Confirm `/home/ubuntu` is visible on compute nodes

```bash
mkdir -p /home/ubuntu/hops_probe
date > /home/ubuntu/hops_probe/from_head.txt
hostname >> /home/ubuntu/hops_probe/from_head.txt
```

```bash
srun -p fast-gpu --nodes=1 --ntasks=1 bash -lc 'hostname; ls -l /home/ubuntu/hops_probe; cat /home/ubuntu/hops_probe/from_head.txt'
```

### 6. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
echo 'source "$HOME/.local/bin/env"' >> ~/.zshrc
uv --version
```

Verify `uv` on a compute node:

```bash
srun -p fast-gpu --nodes=1 --ntasks=1 bash -lc 'source "$HOME/.local/bin/env"; uv --version'
```

### 7. Clone HOPS and Megatron-LM

```bash
cd /home/ubuntu
git clone <hops-repo-url>
git clone https://github.com/NVIDIA/Megatron-LM.git
```

Set up HOPS:

```bash
cd /home/ubuntu/HOPS
source "$HOME/.local/bin/env"
uv sync
uv run pytest -q
uv run python main.py --config configs/default.yaml --no-viz
```

### 8. Build the training environment for Megatron

Use Python 3.12. Megatron did not behave correctly on Python 3.14 in this setup.

```bash
cd /home/ubuntu
source "$HOME/.local/bin/env"
uv python install 3.12
uv venv --python 3.12 megatron-env
source /home/ubuntu/megatron-env/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision torchaudio pyyaml pybind11
cd /home/ubuntu/Megatron-LM
MAX_JOBS=4 uv pip install -e .
```

### 9. Verify PyTorch and distributed launch

Single-node PyTorch GPU check:

```bash
srun -p fast-gpu --nodes=1 --ntasks=1 bash -lc 'source /home/ubuntu/megatron-env/bin/activate; python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"'
```

Two-node visibility check:

```bash
srun -p fast-gpu --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -lc 'source /home/ubuntu/megatron-env/bin/activate; python -c "import torch; print(torch.cuda.get_device_name(0))"'
```

Create the distributed smoke test:

```bash
cat > /home/ubuntu/torch_dist_smoke.py <<'PY'
import os
import socket
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
world = dist.get_world_size()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

x = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(x)

print(
    f"host={socket.gethostname()} rank={rank}/{world} "
    f"local_rank={local_rank} gpu={torch.cuda.get_device_name(local_rank)} sum={x.item()}",
    flush=True,
)

dist.destroy_process_group()
PY
```

Run it inside a two-node allocation:

```bash
salloc -p fast-gpu --nodes=2 --ntasks=2 --ntasks-per-node=1
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
```

```bash
srun --nodes=1 --ntasks=1 -w "${nodes[0]}" bash -lc 'source /home/ubuntu/megatron-env/bin/activate; python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr='"${nodes[0]}"' --master_port=29500 /home/ubuntu/torch_dist_smoke.py' &
srun --nodes=1 --ntasks=1 -w "${nodes[1]}" bash -lc 'source /home/ubuntu/megatron-env/bin/activate; python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr='"${nodes[0]}"' --master_port=29500 /home/ubuntu/torch_dist_smoke.py' &
wait
```

Expected output on both ranks:

- correct hostname
- correct H100 GPU
- `sum=3.0`

### 10. Submit the successful `05_h100_dual_node_pp2` run

The current `experiments/05_h100_dual_node_pp2/run.slurm` already includes the
Megatron compatibility fixes that were needed during debugging:

- `--eval-interval "$TRAIN_ITERS"`
- `--eval-iters 0`
- `--transformer-impl local`
- `--no-persist-layer-norm`
- `--no-gradient-accumulation-fusion`
- `--no-masked-softmax-fusion`

Set the required environment and submit:

```bash
export MEGATRON_DIR=/home/ubuntu/Megatron-LM
export MEGATRON_ENTRYPOINT=pretrain_gpt.py
export TRAIN_ENV_ACTIVATE=/home/ubuntu/megatron-env/bin/activate
```

```bash
cd /home/ubuntu/HOPS
sbatch experiments/05_h100_dual_node_pp2/run.slurm
```

Monitor:

```bash
squeue -u $USER
```

Inspect outputs after completion:

```bash
find /home/ubuntu/HOPS/experiments/05_h100_dual_node_pp2/output/<job_id> -maxdepth 1 -type f | sort
```

Record the submitted Slurm job ID and keep the corresponding `output/<job_id>/`
directory for later calibration.

### 11. Optional communication calibration after the first successful run

To measure cross-node point-to-point timings for the `2 x p5.4xlarge` setup:

```bash
salloc -p fast-gpu --nodes=2 --ntasks=2 --ntasks-per-node=1
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
out=/home/ubuntu/HOPS/experiments/05_h100_dual_node_pp2/output/calibration
mkdir -p "$out"
ts=$(date +%Y%m%d-%H%M%S)
```

```bash
srun --nodes=1 --ntasks=1 -w "${nodes[0]}" bash -lc 'source /home/ubuntu/megatron-env/bin/activate; python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=0 --master_addr='"${nodes[0]}"' --master_port=29540 /home/ubuntu/HOPS/experiments/link_bench.py --mode p2p --sizes-mb 1,4,16,64 --dtype bfloat16 --label h100_pp2' > "$out/p2p_rank0_${ts}.jsonl" &
srun --nodes=1 --ntasks=1 -w "${nodes[1]}" bash -lc 'source /home/ubuntu/megatron-env/bin/activate; python -m torch.distributed.run --nnodes=2 --nproc_per_node=1 --node_rank=1 --master_addr='"${nodes[0]}"' --master_port=29540 /home/ubuntu/HOPS/experiments/link_bench.py --mode p2p --sizes-mb 1,4,16,64 --dtype bfloat16 --label h100_pp2' > "$out/p2p_rank1_${ts}.jsonl" &
wait
```

## Outputs

Each `run.slurm` creates:

- `output/<job_id>/hops_summary.json`
- `output/<job_id>/hops_trace.csv`
- `output/<job_id>/hops_stdout.txt`
- `output/<job_id>/node_inventory.txt`
- one or more Megatron logs

## Quick Output Inspection

From the repo root, get the latest `05_h100_dual_node_pp2` job directory:

```bash
JOB_DIR=$(find experiments/05_h100_dual_node_pp2/output -mindepth 1 -maxdepth 1 -type d | sort -V | tail -1)
echo "$JOB_DIR"
```

List the files in that job output:

```bash
ls "$JOB_DIR"
```

Read the HOPS summary JSON:

```bash
cat "$JOB_DIR/hops_summary.json"
```

Read the simulator stdout summary:

```bash
cat "$JOB_DIR/hops_stdout.txt"
```

Tail the Megatron logs live while a job is running:

```bash
tail -f "$JOB_DIR/megatron_node0.log"
```

```bash
tail -f "$JOB_DIR/megatron_node1.log"
```

Check whether Megatron emitted timing entries:

```bash
rg -n "forward-compute|backward-compute|forward-send|backward-send" \
  "$JOB_DIR"/megatron_node*.log
```

Check for an early launcher or environment failure:

```bash
tail -50 "$JOB_DIR/megatron_node0.log"
tail -50 "$JOB_DIR/megatron_node1.log"
```

Copy the experiment outputs back to a local machine with `rsync`:

```bash
KEY_PATH=<full-path-to-ssh-key.pem>
HEAD_NODE_HOST=<head-node-hostname-or-ip>
REMOTE_OUTPUT_DIR=/home/ubuntu/HOPS/experiments/05_h100_dual_node_pp2/output/
LOCAL_OUTPUT_DIR=/Users/<local-user>/Documents/HOPS/experiments/05_h100_dual_node_pp2/output/

rsync -avz -e "ssh -i $KEY_PATH" \
  "ubuntu@$HEAD_NODE_HOST:$REMOTE_OUTPUT_DIR" \
  "$LOCAL_OUTPUT_DIR"
```

## Plan

Proceed in two phases: first replay reality, then infer from it.

1. Run the two baseline scenarios first.
   Start with `01_h100_baseline_pp2` and `02_a100_baseline_pp2` to measure clean H100-only and A100-only stage behavior.

   If the available cluster shape is instead `2 x p5.4xlarge` (one H100 GPU per node),
   use `05_h100_dual_node_pp2` as the H100 baseline and interpret it as a cross-node PP=2
   reference rather than a same-node NVLink baseline.

2. Measure communication separately.
   Use `link_bench.py` to collect point-to-point and all-reduce timings for the tensor sizes that matter to your pipeline.

3. Keep the first pass in `explicit` mode.
   In this phase, HOPS is validating scheduling, topology, and heterogeneous imbalance while replaying measured stage times rather than trying to infer them.

4. Run the heterogeneous pair.
   Compare `03_hetero_even_pp4` against `04_hetero_weighted_pp4` and check whether the real cluster and HOPS agree on ranking, bottleneck stages, and the approximate gain from weighting the H100 stages more heavily.

5. Replace placeholders with measurements.
   Update the stage distributions in each `hops.yaml` from the Megatron logs, and update `overrides.links` from the communication benchmark.

6. Move to `analytical` mode only after the explicit replay is credible.
   The long-term goal is not to avoid measurement entirely, but to measure enough once that HOPS can predict new heterogeneous layouts with useful accuracy.

## Recommended Next Step After The First Pass

After the first baseline runs, replace the placeholder explicit stage distributions in the `hops.yaml` files with measured means and standard deviations from the Megatron logs. Then rerun the heterogeneous scenarios and compare ranking and error.
