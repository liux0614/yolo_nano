import torch
from torchvision.transforms import (Compose, ToTensor, Normalize, Resize)
from transforms.spatial_transforms import TwoCrop

def compose_transforms(opt):
    transform = Compose([
        TwoCrop
    ])