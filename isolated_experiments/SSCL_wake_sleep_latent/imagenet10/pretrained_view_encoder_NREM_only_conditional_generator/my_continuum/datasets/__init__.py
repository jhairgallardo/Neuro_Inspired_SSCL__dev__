# pylint: disable=C0401
# flake8: noqa
from my_continuum.datasets.base import (
    ImageFolderDataset, InMemoryDataset, PyTorchDataset, _ContinuumDataset, H5Dataset, _AudioDataset
)
from my_continuum.datasets.cifar100 import CIFAR100
from my_continuum.datasets.core50 import (Core50, Core50v2_79, Core50v2_196, Core50v2_391)
from my_continuum.datasets.fellowship import (CIFARFellowship, Fellowship, MNISTFellowship)
from my_continuum.datasets.imagenet import ImageNet100, ImageNet1000, TinyImageNet200
from my_continuum.datasets.synbols import Synbols
from my_continuum.datasets.nlp import MultiNLI
from my_continuum.datasets.pytorch import (
    CIFAR10, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST
)
from my_continuum.datasets.svhn import SVHN
from my_continuum.datasets.colored_mnist import ColoredMNIST
from my_continuum.datasets.rainbow_mnist import RainbowMNIST
from my_continuum.datasets.cub200 import CUB200
from my_continuum.datasets.awa2 import AwA2
from my_continuum.datasets.pascalvoc import PascalVOC2012
from my_continuum.datasets.stream51 import Stream51
from my_continuum.datasets.dtd import DTD
from my_continuum.datasets.vlcs import VLCS
from my_continuum.datasets.pacs import PACS
from my_continuum.datasets.domain_net import DomainNet
from my_continuum.datasets.office_home import OfficeHome
from my_continuum.datasets.terra_incognita import TerraIncognita
from my_continuum.datasets.domain_net import DomainNet
from my_continuum.datasets.rainbow_mnist import RainbowMNIST
from my_continuum.datasets.car196 import Car196
from my_continuum.datasets.caltech import Caltech101, Caltech256
from my_continuum.datasets.fgvc_aircraft import FGVCAircraft
from my_continuum.datasets.stl10 import STL10
from my_continuum.datasets.food101 import Food101
from my_continuum.datasets.omniglot import Omniglot
from my_continuum.datasets.birdsnap import Birdsnap
from my_continuum.datasets.ctrl import CTRL, CTRLplus, CTRLminus, CTRLin, CTRLout, CTRLplastic
from my_continuum.datasets.flowers102 import OxfordFlower102
from my_continuum.datasets.oxford_pet import OxfordPet
from my_continuum.datasets.gtsrb import GTSRB
from my_continuum.datasets.sun397 import SUN397
from my_continuum.datasets.fer2013 import FER2013
from my_continuum.datasets.eurosat import EuroSAT
from my_continuum.datasets.metashift import MetaShift
from my_continuum.datasets.fluentspeech import FluentSpeech
