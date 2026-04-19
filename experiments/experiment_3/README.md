# Experiment 3 Held-Out Generalization Test Set

Experiment 3 is a held-out test set for checking whether HOPS generalizes after
experiment 2 fixtures have been used for calibration or model-fitting. It uses
the same fixed six-GPU cluster as experiment 2: two H100 nodes, two A10G nodes,
and two L4 nodes.

Scenarios `01_*` through `42_*` are the primary Megatron-comparable 1F1B test
set. They cover held-out device orderings, PP2/PP3/PP4/PP5/PP6 depths, unseen
microbatch counts, model shape changes, sequence-length changes, deeper models,
and uneven first/last pipeline layer placement.

Scenarios `43_diag_*` through `48_diag_*` are scheduler diagnostics. Their
Megatron runs still use the current PP run shape, while the HOPS base configs use
GPipe or ZeroBubble scheduler hypotheses. Keep these separate from the clean
held-out 1F1B set when updating validation baselines.

Subset scenarios may allocate more nodes than active pipeline stages and set
`ALLOW_UNUSED_ALLOCATED_NODES=1` in `scenario.env`. With that flag, the common
Slurm helper selects the first unused nodes matching
`EXPECTED_GPU_PATTERN_BY_STAGE`. Default exact-node-count behavior is unchanged
for full-cluster and exact-partition scenarios.

Submit the full chain from the cluster head node:

```bash
cd /home/ubuntu/HOPS
export MEGATRON_DIR=/home/ubuntu/Megatron-LM
export MEGATRON_ENTRYPOINT=pretrain_gpt.py
export TRAIN_ENV_ACTIVATE=/home/ubuntu/megatron-env/bin/activate
bash experiments/experiment_3/submit_sequential.sh
```

For dry runs, write the manifest outside the repo:

```bash
DRY_RUN=1 MANIFEST=/tmp/exp3_sequential.tsv   bash experiments/experiment_3/submit_sequential.sh
```

After runs are copied back locally, extract fixtures and validate:

```bash
uv run python experiments/tools/extract_fixture.py   --experiment experiments/experiment_3   --all

uv run python experiments/tools/validate_fixtures.py
```

Only update golden metrics after confirming the new primary fixtures are useful
and not failed or degenerate runs. Treat `43_diag_*` through `48_diag_*` as
scheduler diagnostics unless explicitly evaluating those scheduler hypotheses.
