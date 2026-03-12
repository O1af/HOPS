"""Latency distributions for random sampling within HOPS."""

import os

RANDOM_SEED = 42

from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):
    """Base class for latency distributions."""

    @abstractmethod
    def sample(self) -> float:
        """Return a non-negative random sample."""

    @classmethod
    def from_yaml(cls, config: dict) -> "Distribution":
        """Factory: build a Distribution from a YAML config dict."""
        registry = {
            "constant": Constant,
            "normal": Normal,
            "heavy_tailed": HeavyTailed,
            "poisson": Poisson,
        }
        kind = config["type"]
        if kind not in registry:
            raise ValueError(f"Unknown distribution type: {kind}")
        params = {k: v for k, v in config.items() if k != "type"}
        return registry[kind](**params)


class Constant(Distribution):
    """Deterministic value — useful for testing."""

    def __init__(self, value: float):
        self.value = value

    def sample(self) -> float:
        return self.value


class Normal(Distribution):
    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std

    def sample(self) -> float:
        return max(0.0, np.random.normal(self.mean, self.std))


class HeavyTailed(Distribution):
    """Pareto distribution for modeling stragglers."""

    def __init__(self, base: float, alpha: float):
        self.base = base
        self.alpha = alpha

    def sample(self) -> float:
        return self.base * np.random.pareto(self.alpha)


class Poisson(Distribution):
    def __init__(self, lam: float):
        self.lam = lam

    def sample(self) -> float:
        return float(np.random.poisson(self.lam))
