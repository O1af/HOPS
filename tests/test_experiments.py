"""Tests grounded in checked-in Megatron/HOPS experiment fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPO_ROOT / "tests/fixtures/experiment_ground_truth.json"


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _throughput_from_iteration_times(iteration_times_ms: list[float], global_batch_size: int) -> float:
    return (len(iteration_times_ms) * global_batch_size) / (sum(iteration_times_ms) / 1000.0)


class TestExperiment05Run26:
    """Ground experiment 05 against extracted Megatron and HOPS artifacts."""

    def test_megatron_log_contains_all_40_iterations(self):
        iterations = _fixture()["exp05_run26"]["megatron_iteration_ms"]

        assert len(iterations) == 40
        assert iterations[0] == pytest.approx(5430.1)
        assert iterations[-1] == pytest.approx(604.5)

    def test_megatron_steady_state_throughput_matches_status_report(self):
        iterations = _fixture()["exp05_run26"]["megatron_iteration_ms"]
        steady_state_times = iterations[2:]

        throughput_per_s = _throughput_from_iteration_times(steady_state_times, global_batch_size=8)

        assert throughput_per_s == pytest.approx(12.4348, rel=1e-4)

    def test_checked_in_hops_baseline_matches_the_run26_artifact(self):
        exp05 = _fixture()["exp05_run26"]

        assert exp05["hops_completed_microbatches"] == 32
        assert exp05["hops_throughput_per_s"] == pytest.approx(26.741269097375586)
        assert exp05["hops_bubble_ratio"] == pytest.approx(0.11659879987806125)

    def test_checked_in_baseline_overpredicts_real_throughput_by_about_115_percent(self):
        exp05 = _fixture()["exp05_run26"]
        steady_state_times = exp05["megatron_iteration_ms"][2:]
        megatron_throughput = _throughput_from_iteration_times(steady_state_times, global_batch_size=8)

        overprediction_pct = (
            (exp05["hops_throughput_per_s"] - megatron_throughput) / megatron_throughput
        ) * 100.0

        assert overprediction_pct == pytest.approx(115.05, abs=0.1)

    def test_fit1_calibration_tracks_run26_megatron_throughput_within_three_percent(self):
        exp05 = _fixture()["exp05_run26"]
        steady_state_times = exp05["megatron_iteration_ms"][2:]
        megatron_throughput = _throughput_from_iteration_times(steady_state_times, global_batch_size=8)

        relative_error = abs(exp05["fit1_throughput_per_s"] - megatron_throughput) / megatron_throughput

        assert relative_error < 0.03


class TestExperiment06Artifacts:
    """Ground experiment 06 against extracted converted Megatron artifacts."""

    def test_run1_trace_includes_the_warmup_heavy_full_window(self):
        durations_ms = _fixture()["exp06_run1"]["megatron_trace_durations_ms"]

        assert len(durations_ms) == 8
        assert durations_ms[0] > 3000.0
        assert durations_ms[1] > 2000.0

    def test_run1_summary_matches_the_checked_in_converted_megatron_output(self):
        exp06_run1 = _fixture()["exp06_run1"]

        assert exp06_run1["megatron_completed_microbatches"] == 4
        assert exp06_run1["megatron_throughput_per_s"] == pytest.approx(0.5978082761117864)
        assert exp06_run1["megatron_bubble_ratio"] == pytest.approx(0.1616322392893241)

    def test_run2_trace_is_a_short_steady_state_capture(self):
        exp06_run2 = _fixture()["exp06_run2"]
        durations_ms = exp06_run2["megatron_trace_durations_ms"]

        assert len(durations_ms) == 4
        assert max(durations_ms) < 8.0
        assert exp06_run2["megatron_completed_microbatches"] == 2
        assert exp06_run2["megatron_throughput_per_s"] == pytest.approx(60.50546751606662)

    def test_run2_comparison_uses_a_shorter_megatron_window_than_the_hops_summary(self):
        exp06_run2 = _fixture()["exp06_run2"]

        assert exp06_run2["megatron_completed_microbatches"] == 2
        assert exp06_run2["hops_completed_microbatches"] == 4
        assert exp06_run2["comparison_megatron_throughput_per_s"] == pytest.approx(
            exp06_run2["megatron_throughput_per_s"]
        )
        assert exp06_run2["comparison_hops_throughput_per_s"] == pytest.approx(33.92759055975943)
