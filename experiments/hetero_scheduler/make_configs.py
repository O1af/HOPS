"""Generate a diverse set of 2×L4 + 2×A10G heterogeneous configs."""

from __future__ import annotations

from itertools import permutations
from pathlib import Path


BASE_A10G = {"id_prefix": "a10g", "gpu": "a10g"}
BASE_L4 = {"id_prefix": "l4", "gpu": "l4"}


def _stage_block(stages: list[str]) -> str:
    out = []
    for i, gpu in enumerate(stages):
        dev = f"{gpu}_n{i}_g0"
        out.append(f"""    - device: {dev}
      weights_mb: 6144
      compute: {{ mode: analytical, tflop: 0.18, memory_mb: 384 }}""")
    return "\n".join(out)


def _hardware_block(stages: list[str]) -> str:
    out = []
    for i, gpu in enumerate(stages):
        out.append(f"    - {{ id: {gpu}_n{i}_g0, gpu: {gpu}, node: {gpu}_n{i}, socket: 0 }}")
    return "\n".join(out)


def _write_config(name: str, stages: list[str], mb: int, hidden: int = 1024,
                  seq: int = 1024, tflop: float = 0.18, mem_mb: float = 384.0,
                  weights_mb: int = 6144) -> None:
    stages_yaml = "\n".join(
        f"""    - device: {gpu}_n{i}_g0
      weights_mb: {weights_mb}
      compute: {{ mode: analytical, tflop: {tflop}, memory_mb: {mem_mb} }}"""
        for i, gpu in enumerate(stages)
    )
    hw_yaml = "\n".join(
        f"    - {{ id: {gpu}_n{i}_g0, gpu: {gpu}, node: {gpu}_n{i}, socket: 0 }}"
        for i, gpu in enumerate(stages)
    )
    body = f"""simulation:
  batches: 1
  microbatches: {mb}
  seed: 42

pipeline:
  schedule: zero_bubble
  precision: bf16
  backward_factor: 2.0
  model:
    hidden_dim: {hidden}
    seq_len: {seq}
  stages:
{stages_yaml}

hardware:
  devices:
{hw_yaml}
  interconnect:
    same_node: pcie
    cross_node: ethernet

optimizer:
  enabled: false

failure:
  enabled: false

output: {{}}
"""
    out = Path(__file__).parent / f"{name}.yaml"
    out.write_text(body)


if __name__ == "__main__":
    configs = [
        ("bench_l4_middle", ["a10g", "l4", "l4", "a10g"], 16),
        ("bench_a10g_middle", ["l4", "a10g", "a10g", "l4"], 16),
        ("bench_alt_al", ["a10g", "l4", "a10g", "l4"], 16),
        ("bench_alt_la", ["l4", "a10g", "l4", "a10g"], 16),
        ("bench_a10g_front", ["a10g", "a10g", "l4", "l4"], 16),
        ("bench_l4_front", ["l4", "l4", "a10g", "a10g"], 16),
        ("bench_l4_middle_mb24", ["a10g", "l4", "l4", "a10g"], 24),
        ("bench_l4_middle_mb48", ["a10g", "l4", "l4", "a10g"], 48),
        ("bench_a10g_middle_mb24", ["l4", "a10g", "a10g", "l4"], 24),
        ("bench_a10g_middle_mb48", ["l4", "a10g", "a10g", "l4"], 48),
        ("bench_big_l4_middle", ["a10g", "l4", "l4", "a10g"], 16, 2048, 1024, 0.70, 512, 8192),
        ("bench_big_a10g_middle", ["l4", "a10g", "a10g", "l4"], 16, 2048, 1024, 0.70, 512, 8192),
    ]
    for cfg in configs:
        _write_config(*cfg)
    print("wrote", len(configs), "configs")
