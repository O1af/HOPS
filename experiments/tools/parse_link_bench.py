"""Convert link_bench.py JSONL output into a HOPS overrides.links fragment.

Input is a directory of JSONL files emitted by experiments/link_bench.py. Each
file contains per-rank rows for a single 2-rank p2p benchmark between an
ordered pair of nodes. We use the sender side (rank 0) as the ground truth for
that pair because the receiver row measures send-initiation latency, not wire
time.

Bandwidth and latency are fit jointly via ordinary least squares on the linear
relation:
    p50_ms(size_mb) = latency_ms + size_mb * (8 / bandwidth_gbps)
across every sampled size for the pair. This is a better estimator than a
single cold large-message p50 — it averages noise and separates the fixed
per-transfer latency from steady-state bandwidth. When only one size was
sampled, we fall back to the legacy single-point method (largest-size p50 for
bandwidth, zero latency).

(size_mb is base-2 MB, so that's MiB*8 bits per ms = Gib/s, which HOPS treats
as Gbps — link_bench.py and HOPS use the same convention.)

The resulting YAML fragment is intended to be consumed by
materialize_hops_variant.py as an overlay on top of hops.base.yaml.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from _common import write_generated_yaml


@dataclass(frozen=True)
class PairMeasurement:
    stage_src: int
    stage_dst: int
    bandwidth_gbps: float
    latency_us: float


def _iter_jsonl_rows(paths: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                # torchrun/NCCL/PyTorch warnings can land in the same file as
                # the benchmark JSON rows. Treat this as mixed log output and
                # keep only the JSON objects emitted by link_bench.py.
                if not stripped.startswith("{"):
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _group_p2p_rows_by_label(rows: list[dict]) -> dict[str, dict[int, list[dict]]]:
    by_label: dict[str, dict[int, list[dict]]] = {}
    for row in rows:
        if row.get("mode") != "p2p":
            continue
        rank = row.get("rank")
        if rank not in (0, 1):
            continue
        label = str(row.get("label", ""))
        by_label.setdefault(label, {0: [], 1: []})[rank].append(row)
    for label, ranks in by_label.items():
        if not ranks[0] or not ranks[1]:
            raise ValueError(
                f"link bench label {label!r} must have both rank 0 and rank 1 rows"
            )
    return by_label


def _stage_pair_from_label(label: str) -> tuple[int, int] | None:
    if not label.startswith("pair_"):
        return None
    parts = label[len("pair_"):].split("_")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _pair_measurement(sender_rows: list[dict]) -> tuple[float, float]:
    by_size = sorted(sender_rows, key=lambda r: int(r["size_mb"]))
    xs = [float(r["size_mb"]) for r in by_size]
    ys = [float(r["p50_ms"]) for r in by_size]
    if not ys or any(y <= 0.0 for y in ys):
        raise ValueError("p50_ms must be positive for bandwidth derivation")

    n = len(xs)
    if n == 1:
        bandwidth_gbps = (xs[0] * 8.0) / ys[0]
        return bandwidth_gbps, 0.0

    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0.0:
        # All sizes identical (degenerate); fall back to single-point.
        bandwidth_gbps = (xs[-1] * 8.0) / ys[-1]
        return bandwidth_gbps, 0.0

    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept_ms = my - slope * mx
    if slope <= 0.0:
        # Noisy fit produced negative or zero slope; fall back to the largest
        # sampled size for bandwidth and clamp latency to the smallest p50.
        bandwidth_gbps = (xs[-1] * 8.0) / ys[-1]
        latency_us = ys[0] * 1000.0
        return bandwidth_gbps, latency_us

    bandwidth_gbps = 8.0 / slope
    latency_us = max(0.0, intercept_ms) * 1000.0
    return bandwidth_gbps, latency_us


def _stage_to_device(trace_map: dict) -> dict[int, str]:
    stages_raw = trace_map.get("stages")
    if not isinstance(stages_raw, dict):
        raise ValueError("trace map must contain a 'stages' object")
    return {int(stage_id): str(device) for stage_id, device in stages_raw.items()}


def derive_pair_measurements(
    rows: list[dict],
    label_to_stages: dict[str, tuple[int, int]] | None = None,
) -> list[PairMeasurement]:
    by_label = _group_p2p_rows_by_label(rows)

    measurements: list[PairMeasurement] = []
    for label in sorted(by_label):
        stages = (label_to_stages or {}).get(label) or _stage_pair_from_label(label)
        if stages is None:
            raise ValueError(
                f"Cannot infer stage pair from label {label!r}; expected 'pair_<i>_<j>' "
                "or pass an explicit label_to_stages map"
            )
        bandwidth_gbps, latency_us = _pair_measurement(by_label[label][0])
        measurements.append(
            PairMeasurement(
                stage_src=stages[0],
                stage_dst=stages[1],
                bandwidth_gbps=bandwidth_gbps,
                latency_us=latency_us,
            )
        )
    return measurements


def build_overrides_yaml(
    measurements: list[PairMeasurement],
    stage_to_device: dict[int, str],
) -> dict:
    links: list[dict] = []
    for m in measurements:
        if m.stage_src not in stage_to_device or m.stage_dst not in stage_to_device:
            raise ValueError(
                f"trace map is missing device entry for stages "
                f"{m.stage_src} or {m.stage_dst}"
            )
        src_device = stage_to_device[m.stage_src]
        dst_device = stage_to_device[m.stage_dst]
        common = {
            "bandwidth_gbps": round(m.bandwidth_gbps, 4),
            "latency_us": round(m.latency_us, 4),
        }
        links.append({"src": src_device, "dst": dst_device, **common})
        links.append({"src": dst_device, "dst": src_device, **common})
    return {"overrides": {"links": links}}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing link_bench JSONL files (one per adjacent pipeline pair)",
    )
    parser.add_argument(
        "--trace-map",
        required=True,
        help="Path to hops_trace_map.json produced by run.slurm (maps stage id to device id)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the generated links.yaml overlay",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"link bench input dir not found: {input_dir}")
    jsonl_paths = sorted(input_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise SystemExit(f"no *.jsonl files in {input_dir}")

    trace_map = json.loads(Path(args.trace_map).read_text(encoding="utf-8"))
    stage_to_device = _stage_to_device(trace_map)

    rows = _iter_jsonl_rows(jsonl_paths)
    measurements = derive_pair_measurements(rows)
    document = build_overrides_yaml(measurements, stage_to_device)

    banner = (
        "# GENERATED by experiments/tools/parse_link_bench.py. DO NOT EDIT.\n"
        "# Source: measured link_bench p2p timings.\n"
        "# Bandwidth and latency jointly fit by OLS on p50_ms vs size_mb.\n"
    )
    output_path = Path(args.output)
    write_generated_yaml(output_path, banner, document)

    print(f"wrote {len(measurements)} pair measurements -> {output_path}")


if __name__ == "__main__":
    main()
