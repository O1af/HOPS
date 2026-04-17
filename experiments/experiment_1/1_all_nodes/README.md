# Experiment 1: All Nodes

This scenario uses all four G-family On-Demand Capacity Reservations in
`us-east-1f` as one Slurm partition:

- `g5a`: `g5.2xlarge`, ODCR `cr-086df7f9e6889625e`
- `g5b`: `g5.2xlarge`, ODCR `cr-09c1aed639199e8b1`
- `g6a`: `g6.4xlarge`, ODCR `cr-035a73de7465afe91`
- `g6b`: `g6.4xlarge`, ODCR `cr-0effa19780678c103`

Create the shared experiment cluster:

```bash
pcluster create-cluster \
  --cluster-name hops-g-experiment-1 \
  --cluster-configuration experiments/experiment_1/cluster_g5_g6_odcr_1f.yaml \
  --region us-east-1
```

Submit the all-node run from the cluster head node:

```bash
export MEGATRON_DIR=/home/ubuntu/Megatron-LM
export TRAIN_ENV_ACTIVATE=/home/ubuntu/megatron-env/bin/activate
sbatch experiments/experiment_1/1_all_nodes/run.slurm
```

This first pass uses analytical compute estimates in `hops.yaml` rather than
explicit placeholder timing distributions. It also omits `activation_mb`; HOPS
derives the pipeline activation size from the `pipeline.model` block so it stays
aligned with the Megatron shape in `run.slurm`.

After the first successful real run, replace the analytical estimates with
measured explicit stage distributions if you need calibrated replay mode.
