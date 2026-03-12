"""
Distributions used for random sampling within HOPS

Abstract Distribution type contains a sample method.
Supported distributions include:

   - Normal
   - Heavey-Tailed
   - Poisson
   - TODO
"""

import os

RANDOM_SEED = 42

from abc import ABC, abstractmethod

class Distribution(ABC):
   "TODO docstring"

   @abstractmethod
   def sample(self) -> float:
      "TODO dosctring"

class Normal(Distribution):

   def sample(self) -> float:
      "TODO dosctring"
      return 4.0 #TODO

class HeavyTailed(Distribution):
   pass

class Poisson(Distribution):
   pass
