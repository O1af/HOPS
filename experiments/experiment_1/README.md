# Experiment 1

Experiment 1 tracks AWS G-family heterogeneous validation runs.

The first scenario, `1_all_nodes`, uses all four reserved
On-Demand G nodes in `us-east-1f`:

- 2 x `g5.2xlarge` = 2 x NVIDIA A10G
- 2 x `g6.4xlarge` = 2 x NVIDIA L4

Later `1.x` scenarios can reuse the same cluster shape while narrowing the
allocation to subsets of the G5 and G6 nodes.

Shared cluster config:

```bash
pcluster create-cluster \
  --cluster-name hops-g-experiment-1 \
  --cluster-configuration experiments/experiment_1/cluster_g5_g6_odcr_1f.yaml \
  --region us-east-1
```
