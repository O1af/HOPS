"""Convert link_bench.py JSONL output into a HOPS overrides.links fragment.

Input is a directory of JSONL files emitted by experiments/link_bench.py. Each
file contains per-rank rows for a single 2-rank p2p benchmark between an
ordered pair of nodes. We use the sender side (rank 0) as the ground truth for
that pair because the receiver row measures send-initiation latency, not wire
time.

Bandwidth is derived from the largest sampled size:
    bandwidth_gbps = size_mb * 8 / p50_ms
(size_mb is base-2 MB, so that's MiB*8 bits per ms = Gib/s, which HOPS treats
as Gbps — link_bench.py and HOPS use the same convention.)

Latency is taken from the smallest sampled size's p50_ms, converted to us.

The resulting YAML fragment is intended to be consumed by
materialize_hops_variant.py as an overlay on top of hops.base.yaml.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


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
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _sender_rows_by_pair(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group rank-0 (sender) rows by their (label, hostname) pair identity.

    The benchmark is launched per adjacent pipeline pair with a distinct label
    like 'pair_0_1'. We key on the label so multiple pair files can co-exist
    in one input directory without interleaving.
    """
    by_label: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("mode") != "p2p":
            continue
        label = str(row.get("label", ""))
        by_label.setdefault(label, []).append(row)

    grouped: dict[tuple[str, str], list[dict]] = {}
    for label, label_rows in by_label.items():
        sender = [r for r in label_rows if r.get("rank") == 0]
        receiver = [r for r in label_rows if r.get("rank") == 1]
        if not sender or not receiver:
            raise ValueError(
                f"link bench label {label!r} must have both rank 0 and rank 1 rows"
            )
        src_host = str(sender[0]["hostname"])
        dst_host = str(receiver[0]["hostname"])
        grouped[(src_host, dst_host)] = sender
    return grouped


def _stage_pair_from_label(label: str) -> tuple[int, int] | None:
    """Extract the adjacent stage pair encoded in a label like 'pair_0_1'."""
    if not label.startswith("pair_"):
        return None
    tail = label[len("pair_"):]
    parts = tail.split("_")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _pair_measurement(sender_rows: list[dict]) -> tuple[float, float]:
    by_size = sorted(sender_rows, key=lambda r: int(r["size_mb"]))
    smallest = by_size[0]
    largest = by_size[-1]

    latency_us = float(smallest["p50_ms"]) * 1000.0

    size_mb = float(largest["size_mb"])
    time_ms = float(largest["p50_ms"])
    if time_ms <= 0.0:
        raise ValueError("p50_ms must be positive for bandwidth derivation")
    bandwidth_gbps = (size_mb * 8.0) / time_ms
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
    grouped = _sender_rows_by_pair(rows)

    # Rebuild label -> sender rows so we can pair labels with stage indices.
    label_rows: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("mode") != "p2p" or row.get("rank") != 0:
            continue
        label_rows.setdefault(str(row.get("label", "")), []).append(row)

    measurements: list[PairMeasurement] = []
    for label, sender_rows in sorted(label_rows.items()):
        stages = (label_to_stages or {}).get(label) or _stage_pair_from_label(label)
        if stages is None:
            raise ValueError(
                f"Cannot infer stage pair from label {label!r}; expected 'pair_<i>_<j>' "
                "or pass an explicit label_to_stages map"
            )
        bandwidth_gbps, latency_us = _pair_measurement(sender_rows)
        measurements.append(
            PairMeasurement(
                stage_src=stages[0],
                stage_dst=stages[1],
                bandwidth_gbps=bandwidth_gbps,
                latency_us=latency_us,
            )
        )
    # grouped is validated above; unused locally but ensures receiver rows exist.
    del grouped
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    banner = (
        "# GENERATED by experiments/tools/parse_link_bench.py. DO NOT EDIT.\n"
        "# Source: measured link_bench p2p timings.\n"
        "# Bandwidth from largest sampled size p50; latency from smallest size p50.\n"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(banner)
        yaml.safe_dump(document, handle, sort_keys=False)

    print(f"wrote {len(measurements)} pair measurements -> {output_path}")


if __name__ == "__main__":
    main()
