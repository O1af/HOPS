# Experiment 2 Scenario Matrix

Experiment 2 runs against the fixed six-GPU cluster described by
`cluster_g5_g6_p5_6gpu_1f.yaml`: two H100 nodes, two A10G nodes, and two L4
nodes. Scenarios `1_*` through `12_*` are the original validation matrix.
Scenarios `13_*` through `36_*` expand the matrix for fixture collection.

The added scenarios cover:

- homogeneous PP2 pairs for A10G and L4,
- mixed PP2, PP3, and PP5 device orderings,
- H100-only and G-family-only schedule and microbatch variants,
- full-cluster microbatch, GPipe, and ZeroBubble variants,
- short-sequence, long-sequence, wider-model, and uneven-layer cases.

Subset scenarios that need only part of an allocated partition set
`ALLOW_UNUSED_ALLOCATED_NODES=1` in `scenario.env`. With that flag, the common
Slurm helper may allocate more nodes than pipeline stages and select the first
unused nodes matching `EXPECTED_GPU_PATTERN_BY_STAGE`. Without the flag, the
existing exact-node-count behavior is preserved.

Submit the full chain from the cluster head node:

```bash
cd /home/ubuntu/HOPS
export MEGATRON_DIR=/home/ubuntu/Megatron-LM
export MEGATRON_ENTRYPOINT=pretrain_gpt.py
export TRAIN_ENV_ACTIVATE=/home/ubuntu/megatron-env/bin/activate
bash experiments/experiment_2/submit_sequential.sh
```

After runs are copied back locally, extract fixtures and validate:

```bash
uv run python experiments/tools/extract_fixture.py \
  --experiment experiments/experiment_2 \
  --all

uv run python experiments/tools/validate_fixtures.py
```

Only update `fixtures/expected_metrics.json` after confirming the new fixtures
are useful and not failed or degenerate runs:

```bash
uv run python experiments/tools/validate_fixtures.py --update-golden
```
