# pylint: disable=C0401
# flake8: noqa
from my_continuum.scenarios.base import _BaseScenario
from my_continuum.scenarios.continual_scenario import ContinualScenario
from my_continuum.scenarios.class_incremental import ClassIncremental
from my_continuum.scenarios.instance_incremental import InstanceIncremental
from my_continuum.scenarios.specific_scenarios import ALMA
from my_continuum.scenarios.transformation_incremental import TransformationIncremental
from my_continuum.scenarios.rotations import Rotations
from my_continuum.scenarios.permutations import Permutations
from my_continuum.scenarios.segmentation import SegmentationClassIncremental
from my_continuum.scenarios.hashed import HashedScenario
from my_continuum.scenarios.online_fellowship import OnlineFellowship
from my_continuum.scenarios import hf

from my_continuum.scenarios.scenario_utils import create_subscenario, encode_scenario, remap_class_vector, get_scenario_remapping

__all__ = [
    "ContinualScenario",
    "ClassIncremental",
    "InstanceIncremental",
    "Rotations",
    "Permutations",
    "TransformationIncremental",
    "SegmentationClassIncremental",
    "HashedScenario",
    "OnlineFellowship",
    "hf",
    "ALMA"
]
