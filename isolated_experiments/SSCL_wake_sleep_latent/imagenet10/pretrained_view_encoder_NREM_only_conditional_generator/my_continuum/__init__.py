"""Continuum lib.

Made by Arthur Douillard and Timothee Lesort.

The goal of this library is to provide clean and simple to use utilities for
Continual Learning.
"""
# pylint: disable=C0401
# flake8: noqa
from my_continuum import datasets
from my_continuum.scenarios import *
from my_continuum.tasks import *
from my_continuum.metrics import *
from my_continuum.viz import plot_samples
from my_continuum import generators
from my_continuum import rehearsal
from my_continuum import transforms

__version__ = "1.2.4"
