"""
Algorithms package for continual learning playground.
"""

from .ewc import EWC
from .replay import ExperienceReplay
from .naive import NaiveLearning

__all__ = ['EWC', 'ExperienceReplay', 'NaiveLearning']
