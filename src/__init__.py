######################################################################
# Init file for the package
######################################################################
# It designates a directory as a Python package, allowing its modules to be imported elsewhere in your project.

__version__ = "v.0.0"
__author__ = "Alessandro Venaruzzo"

# Libraries import

import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os