"""Tests for latency distributions."""

import numpy as np
import pytest

from hops.latency.distributions import (
    Constant,
    Distribution,
    HeavyTailed,
    Normal,
    Poisson,
)


def test_constant_returns_fixed_value():
    rng = np.random.default_rng(0)
    d = Constant(3.14)
    assert d.sample(rng) == 3.14
    assert d.sample(rng) == 3.14


def test_normal_mean_within_tolerance():
    rng = np.random.default_rng(0)
    d = Normal(mean=10.0, std=1.0)
    samples = [d.sample(rng) for _ in range(10000)]
    assert abs(np.mean(samples) - 10.0) < 0.1


def test_normal_never_negative():
    rng = np.random.default_rng(0)
    d = Normal(mean=0.1, std=5.0)
    samples = [d.sample(rng) for _ in range(1000)]
    assert all(s >= 0.0 for s in samples)


def test_heavy_tailed_produces_outliers():
    rng = np.random.default_rng(42)
    d = HeavyTailed(base=1.0, alpha=2.0)
    samples = [d.sample(rng) for _ in range(10000)]
    assert max(samples) > 5.0


def test_heavy_tailed_respects_base_floor():
    rng = np.random.default_rng(0)
    d = HeavyTailed(base=6.0, alpha=2.5)
    samples = [d.sample(rng) for _ in range(1000)]
    assert min(samples) >= 6.0


def test_poisson_mean_within_tolerance():
    rng = np.random.default_rng(0)
    d = Poisson(lam=5.0)
    samples = [d.sample(rng) for _ in range(10000)]
    assert abs(np.mean(samples) - 5.0) < 0.2


def test_from_yaml_normal():
    d = Distribution.from_yaml({"type": "normal", "mean": 5.0, "std": 1.0})
    assert isinstance(d, Normal)


def test_from_yaml_constant():
    rng = np.random.default_rng(0)
    d = Distribution.from_yaml({"type": "constant", "value": 7.0})
    assert isinstance(d, Constant)
    assert d.sample(rng) == 7.0


def test_from_yaml_heavy_tailed():
    d = Distribution.from_yaml({"type": "heavy_tailed", "base": 1.0, "alpha": 2.0})
    assert isinstance(d, HeavyTailed)


def test_from_yaml_unknown_raises():
    with pytest.raises(ValueError):
        Distribution.from_yaml({"type": "unknown"})
