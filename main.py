import argparse

import numpy as np
import yaml

from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import make_scheduler
from hops.failure.engine import FailureEngine
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.metrics.collector import MetricsCollector
from hops.metrics.reporter import Reporter
from hops.viz.dashboard import draw_dashboard
from hops.viz.timeline import draw_timeline


def main():
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
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    np.random.seed(config["simulation"]["seed"])

    # Build components
    topology = Topology.from_yaml(config["hardware"])
    compute_model = ComputeModel.from_yaml(config["pipeline"])
    scheduler = make_scheduler(config["scheduler"])
    collector = MetricsCollector()
    engine = EventEngine()

    stages = [
        Stage(id=s["id"], device_id=s["device"]) for s in config["pipeline"]["stages"]
    ]

    activation_size_mb = config["hardware"].get("activation_size_mb", 50.0)
    pipeline = Pipeline(
        stages,
        engine,
        topology,
        compute_model,
        scheduler,
        collector,
        activation_size_mb,
    )

    # Optional failure injection
    if config.get("failure", {}).get("enabled", False):
        pipeline.set_failure_engine(
            FailureEngine(engine, topology, collector, config["failure"])
        )

    # Run simulation
    num_batches = config["simulation"]["num_batches"]
    num_microbatches = config["simulation"]["num_microbatches"]

    for batch_idx in range(num_batches):
        pipeline.start_batch(num_microbatches)
        engine.run(stop_condition=lambda: pipeline.batch_complete)

    # Report
    Reporter(collector).print_summary()

    # Visualize
    if not args.no_viz:
        output_cfg = config.get("output", {})
        draw_timeline(collector, output_cfg.get("timeline_path", "output/timeline.png"))
        draw_dashboard(
            collector, output_cfg.get("dashboard_path", "output/dashboard.png")
        )


if __name__ == "__main__":
    main()
