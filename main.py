import argparse

import numpy as np
import yaml

from hops.core.event_engine import EventEngine
from hops.core.pipeline import Pipeline, Stage
from hops.core.scheduler import make_scheduler, max_in_flight_count
from hops.core.types import AllreduceAlgo, Precision
from hops.failure.engine import FailureEngine
from hops.hardware.topology import Topology
from hops.latency.compute_model import ComputeModel
from hops.latency.distributions import Distribution
from hops.metrics.collector import MetricsCollector
from hops.metrics.reporter import Reporter


def validate_stage_layout(topology, stage_configs) -> None:
    """Validate stage IDs, devices, and required neighbor links."""
    if not stage_configs:
        raise ValueError("Pipeline must define at least one stage")

    stage_ids = [stage["id"] for stage in stage_configs]
    if len(set(stage_ids)) != len(stage_ids):
        raise ValueError(f"Duplicate stage IDs are not allowed: {stage_ids}")

    expected_ids = list(range(len(stage_configs)))
    if stage_ids != expected_ids:
        raise ValueError(
            "Stage IDs must be contiguous zero-based integers in pipeline order. "
            f"Expected {expected_ids}, got {stage_ids}"
        )

    for stage in stage_configs:
        device_id = stage["device"]
        try:
            topology.device(device_id)
        except KeyError as exc:
            raise ValueError(
                f"Stage {stage['id']} references unknown device {device_id!r}"
            ) from exc

    for left, right in zip(stage_configs, stage_configs[1:]):
        src = left["device"]
        dst = right["device"]
        try:
            topology.link(src, dst)
        except KeyError as exc:
            raise ValueError(
                f"Pipeline forward transfer requires a link from {src!r} to {dst!r}"
            ) from exc
        try:
            topology.link(dst, src)
        except KeyError as exc:
            raise ValueError(
                f"Pipeline backward transfer requires a link from {dst!r} to {src!r}"
            ) from exc


def validate_memory(topology, stage_configs, policy: str,
                    num_microbatches: int, activation_size_mb: float,
                    precision: Precision) -> None:
    """Check a conservative per-device peak memory bound."""
    eff_activation = activation_size_mb * precision.data_scale
    num_stages = len(stage_configs)
    weight_overhead = precision.weight_memory_overhead
    usage_by_device: dict[str, dict[str, float]] = {}

    for i, stage_cfg in enumerate(stage_configs):
        device = topology.device(stage_cfg["device"])
        usage = usage_by_device.setdefault(
            device.id, {"weights_mb": 0.0, "activations_mb": 0.0}
        )
        usage["weights_mb"] += stage_cfg.get("memory_mb", 0.0) * weight_overhead
        usage["activations_mb"] += (
            eff_activation
            * max_in_flight_count(policy, i, num_stages, num_microbatches)
        )

    for device_id, usage in usage_by_device.items():
        device = topology.device(device_id)
        peak = usage["weights_mb"] + usage["activations_mb"]
        if peak > device.memory_mb:
            raise ValueError(
                f"Device {device.id}: peak memory {peak:.1f} MB exceeds device "
                f"capacity {device.memory_mb:.1f} MB "
                f"(weights={usage['weights_mb']:.1f} MB, "
                f"activations={usage['activations_mb']:.1f} MB)"
            )


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
    stage_configs = config["pipeline"]["stages"]
    validate_stage_layout(topology, stage_configs)

    precision = Precision(config.get("pipeline", {}).get("precision", "fp32"))
    precision_speedup = precision.compute_speedup

    compute_model = ComputeModel.from_yaml({
        **config["pipeline"],
        "precision_speedup": precision_speedup,
    })
    scheduler = make_scheduler(config["scheduler"])
    collector = MetricsCollector()
    engine = EventEngine()

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
    allreduce_algo = AllreduceAlgo(opt_cfg.get("allreduce_algo", "naive"))
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
        draw_timeline, draw_dashboard = load_visualizers()
        output_cfg = config.get("output", {})
        draw_timeline(collector, output_cfg.get("timeline_path", "output/timeline.png"))
        draw_dashboard(
            collector, output_cfg.get("dashboard_path", "output/dashboard.png")
        )


if __name__ == "__main__":
    main()
