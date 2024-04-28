import os
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/data/datasets/ImageNet-10', help='Directory containing the data')


def main():

    # Load encoder (get feature map 512x7x7 as output)
    encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
    encoder.fc = torch.nn.Identity()
    encoder.avgpool = torch.nn.Identity()

    # # Load decoder (get 3x224x224 as output) (simple stacked transposedconvolutions)
    # decoder = torch.nn.Sequential(
    #     torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2), #  kernel_size=4, padding=1
    #     torch.nn.ReLU(),
    #     torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
    #     torch.nn.ReLU(),
    #     torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
    #     torch.nn.ReLU(),
    #     torch.nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
    #     torch.nn.Sigmoid()
    # )

    # Load decoder (get 3x224x224 as output) (simple using upsampling nearest and convolutions)
    decoder = torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2, mode='nearest'),
        torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2, mode='nearest'),
        torch.nn.Conv2d(256, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2, mode='nearest'),
        torch.nn.Conv2d(128, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Upsample(scale_factor=2, mode='nearest'),
        torch.nn.Conv2d(64, 3, kernel_size=3, padding=1),
        torch.nn.Sigmoid()
    )

    # Load data
    # TODO

    



    return None


if __name__ == '__main__':
    main()
