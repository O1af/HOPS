import argparse

import numpy as np
import yaml

from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import make_scheduler, max_in_flight_count
from hops.failure.engine import FailureEngine
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Distribution
from hops.metrics.collector import MetricsCollector
from hops.metrics.reporter import Reporter
from hops.viz.dashboard import draw_dashboard
from hops.viz.timeline import draw_timeline


def validate_memory(topology, stage_configs, policy: str,
                    num_microbatches: int, activation_size_mb: float,
                    precision: str) -> None:
    """Check that peak memory fits on each device. Raises ValueError if not."""
    scale = 0.5 if precision in ("fp16", "bf16") else 1.0
    eff_activation = activation_size_mb * scale
    num_stages = len(stage_configs)

    for i, stage_cfg in enumerate(stage_configs):
        device = topology.device(stage_cfg["device"])
        weight_mem = stage_cfg.get("memory_mb", 0.0)
        weight_overhead = 1.5 if precision in ("fp16", "bf16") else 1.0
        max_activations = max_in_flight_count(policy, i, num_stages, num_microbatches)
        peak = weight_mem * weight_overhead + eff_activation * max_activations
        if peak > device.memory_mb:
            raise ValueError(
                f"Stage {stage_cfg['id']} on {device.id}: peak memory {peak:.1f} MB "
                f"exceeds device capacity {device.memory_mb:.1f} MB "
                f"(weights={weight_mem * weight_overhead:.1f} MB, "
                f"activations={eff_activation:.1f}x{max_activations}={eff_activation * max_activations:.1f} MB)"
            )


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

    rng = np.random.default_rng(config["simulation"]["seed"])

    # Build components
    topology = Topology.from_yaml(config["hardware"])

    precision = config.get("pipeline", {}).get("precision", "fp32")
    precision_speedup = {"fp32": 1.0, "fp16": 2.0, "bf16": 2.0}.get(precision, 1.0)

    compute_model = ComputeModel.from_yaml({
        **config["pipeline"],
        "precision_speedup": precision_speedup,
    })
    scheduler = make_scheduler(config["scheduler"])
    collector = MetricsCollector()
    engine = EventEngine()

    stage_configs = config["pipeline"]["stages"]
    stages = [
        Stage(id=s["id"], device_id=s["device"]) for s in stage_configs
    ]

    activation_size_mb = config["hardware"].get("activation_size_mb", 50.0)
    num_microbatches = config["simulation"]["num_microbatches"]
    policy = config["scheduler"]["policy"]

    # Memory validation
    validate_memory(topology, stage_configs, policy, num_microbatches,
                    activation_size_mb, precision)

    # Optional optimizer step
    opt_cfg = config.get("optimizer", {})
    optimizer_latency = None
    gradient_size_mb = 0.0
    gradient_accumulation_steps = opt_cfg.get("gradient_accumulation_steps", 1)
    allreduce_algo = opt_cfg.get("allreduce_algo", "naive")
    if opt_cfg.get("enabled", False):
        optimizer_latency = Distribution.from_yaml(opt_cfg["compute_latency"])
        gradient_size_mb = opt_cfg.get("gradient_size_mb", 0.0)

    stage_memory_mb = {s["id"]: s.get("memory_mb", 0.0) for s in stage_configs}

    pipeline = Pipeline(
        stages,
        engine,
        topology,
        compute_model,
        scheduler,
        collector,
        activation_size_mb,
        rng=rng,
        optimizer_latency=optimizer_latency,
        gradient_size_mb=gradient_size_mb,
        stage_memory_mb=stage_memory_mb,
        gradient_accumulation_steps=gradient_accumulation_steps,
        precision=precision,
        allreduce_algo=allreduce_algo,
    )

    # Optional failure injection
    if config.get("failure", {}).get("enabled", False):
        pipeline.set_failure_engine(
            FailureEngine(engine, topology, collector, config["failure"], rng=rng)
        )

    # Run simulation
    num_batches = config["simulation"]["num_batches"]

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
