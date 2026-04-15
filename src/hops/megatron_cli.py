"""CLI for converting Megatron raw traces into HOPS-compatible outputs."""

from __future__ import annotations

import argparse

from hops.megatron.compare import convert_job_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw Megatron traces into HOPS-compatible artifacts"
    )
    parser.add_argument(
        "--job-dir",
        required=True,
        help="Experiment output directory containing megatron_trace/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = convert_job_dir(args.job_dir)
    print(f"Wrote {outputs.megatron_trace_csv}")
    print(f"Wrote {outputs.megatron_summary_json}")
    if outputs.comparison_json is not None:
        print(f"Wrote {outputs.comparison_json}")


if __name__ == "__main__":
    main()
