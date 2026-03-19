"""CLI entrypoint for HOPS."""

from __future__ import annotations

import argparse

import yaml

from hops.config import parse_config
from hops.metrics.exporter import TraceExporter
from hops.runtime import build_runtime


def load_visualizers():
    """Import visualization helpers only when plotting is requested."""
    try:
        from hops.viz.dashboard import draw_dashboard
        from hops.viz.timeline import draw_timeline
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("matplotlib"):
            raise RuntimeError(
                "Visualization requires matplotlib. Install the dev dependencies "
                "or rerun with --no-viz."
            ) from exc
        raise
    return draw_timeline, draw_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HOPS: Heterogeneous Optimized Pipeline Simulator"
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--no-viz", action="store_true", help="Skip visualization output"
    )
    parser.add_argument(
        "--summary-json", help="Write machine-readable summary metrics to JSON"
    )
    parser.add_argument(
        "--trace-csv", help="Write compute/transfer/failure trace rows to CSV"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)

    config = parse_config(raw_config)
    runtime = build_runtime(config)

    for _ in range(runtime.num_batches):
        runtime.pipeline.start_batch(runtime.num_microbatches)
        runtime.engine.run(stop_condition=lambda: runtime.pipeline.batch_complete)

    runtime.reporter.print_summary()

    summary_json_path = args.summary_json or runtime.output_config.summary_json
    if summary_json_path:
        runtime.reporter.write_summary_json(summary_json_path)

    trace_csv_path = args.trace_csv or runtime.output_config.trace_csv
    if trace_csv_path:
        TraceExporter(runtime.collector).write_csv(trace_csv_path)

    if not args.no_viz:
        draw_timeline, draw_dashboard = load_visualizers()
        if runtime.output_config.timeline:
            draw_timeline(runtime.collector, runtime.output_config.timeline)
        if runtime.output_config.dashboard:
            draw_dashboard(
                runtime.reporter.summary_model(),
                runtime.reporter.analyzer.e2e_latencies(),
                runtime.output_config.dashboard,
            )


if __name__ == "__main__":
    main()
