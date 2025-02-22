from __future__ import print_function
import argparse
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
# import h5py
# import faiss

# from tensorboardX import SummaryWriter
import numpy as np

import netvlad


def cosine_similarity(v1, v2):
    """Compute the cosine similarity between two VLAD encodings"""
    v1_norm = v1 / torch.linalg.norm(v1)
    v2_norm = v2 / torch.linalg.norm(v2)
    return torch.dot(v1_norm, v2_norm).item()

def euclidean_distance(v1, v2):
    return torch.dist(v1, v2, p=2).item()


print(f'Is cuda available? {torch.cuda.is_available()}')    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG16 as Encoder
encoder_dim = 512  # VGG16 output dimension
encoder = models.vgg16(pretrained=False)
layers = list(encoder.features.children())[:-2]  # Use the feature extraction layers of VGG16
encoder = nn.Sequential(*layers)
encoder = encoder.to(device)

# Load NetVLAD layer as pooling
net_vlad = netvlad.NetVLAD(num_clusters=64, dim=encoder_dim, vladv2=False).to(device)

# Load the checkpoint
checkpoint_path = 'vgg16_netvlad_checkpoint/checkpoints/checkpoint.pth.tar'  # Update this path with your actual checkpoint
# checkpoint = torch.load(checkpoint_path, map_location=device)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only = False)


# Separate encoder and net_vlad weights
encoder_state_dict = {k.replace('encoder.', ''): v for k, v in checkpoint['state_dict'].items() if 'encoder' in k}
netvlad_state_dict = {k.replace('pool.', ''): v for k, v in checkpoint['state_dict'].items() if 'pool' in k}

# Load the encoder state dictionary
missing_encoder_keys, unexpected_encoder_keys = encoder.load_state_dict(encoder_state_dict, strict=False)
print("Encoder - Missing keys:", missing_encoder_keys)
print("Encoder - Unexpected keys:", unexpected_encoder_keys)

# Load the NetVLAD state dictionary
missing_vlad_keys, unexpected_vlad_keys = net_vlad.load_state_dict(netvlad_state_dict, strict=False)
print("NetVLAD - Missing keys:", missing_vlad_keys)
print("NetVLAD - Unexpected keys:", unexpected_vlad_keys)

# Define the transformation for input image
#Resize to 480 * 640 to comply ImageNet dataset

#Resize, Normalize
transform = transforms.Compose([
    transforms.Resize((480, 640)),  # Resize to the size used during training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet values
                         std=[0.229, 0.224, 0.225])
])

#No resize, normalize
uncompress_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet values
                         std=[0.229, 0.224, 0.225])
])

# Define the transformation for input image
#But without normalization
#For simplicity to construct adversarial patch

#Resize, no normalize
unnormalized_transform = transforms.Compose([
    #we need customized resize (960,432)
    transforms.Resize((432, 960)),  # Resize to the size used during training
#     transforms.Resize((480, 640)),  # Resize to the size used during training
    transforms.ToTensor()
])

#No resize, no normalize
uncompress_unnormalized_transform = transforms.Compose([
    transforms.ToTensor()
])

# Define the transformation for input image
#But without normalization
#For simplicity to construct adversarial patch

#Resize, no normalize
unnormalized_transform = transforms.Compose([
    #we need customized resize (960,432)
    transforms.Resize((432, 960)),  # Resize to the size used during training
#     transforms.Resize((480, 640)),  # Resize to the size used during training
    transforms.ToTensor()
])

#No resize, no normalize
uncompress_unnormalized_transform = transforms.Compose([
    transforms.ToTensor()
])

#Extra helper function for tensor -> encodings
#Non-normalzied version
def image_to_tensor(image_path, compress = True, normalize = True):
  image = Image.open(image_path).convert('RGB')

  if compress and not normalize:
    image = unnormalized_transform(image).unsqueeze(0)  # Add batch dimension
  elif not compress and not normalize:
    image = uncompress_unnormalized_transform(image).unsqueeze(0)  # Add batch dimension
  elif compress and normalize:
    image = transform(image).unsqueeze(0)
  elif not compress and normalize:
    image = uncompress_transform(image).unsqueeze(0)
  return image #which is a cpu tensor of torch.Size([1, 3, 480, 640])
# Freeze the encoder parameters (VGG16)
for param in encoder.parameters():
    param.requires_grad = False

# Optionally, freeze the NetVLAD parameters as well (if needed)
for param in net_vlad.parameters():
    param.requires_grad = False

#Image tensor to encodings
def extract_vlad_encoding_tensor(image_tensor):
    image_tensor = image_tensor.to(device)

    # Pass the image through the VGG16 + NetVLAD model
    encoder.eval()
    net_vlad.eval()

    # Pass the image through the encoder (VGG16)
    image_encoding = encoder(image_tensor)
    # Pass the encoded image through NetVLAD pooling
    vlad_encoding = net_vlad(image_encoding)
    # print('The shape of vlad encoding is: ', vlad_encoding.shape)

    return vlad_encoding

#Image file path to encodings
def extract_vlad_encoding_path(image_path, compress = True, normalize = True):
    # Load and preprocess the image
    image_tensor = image_to_tensor(image_path, compress, normalize)
    image_tensor = image_tensor.to(device)

    # Pass the image through the VGG16 + NetVLAD model
    vlad_encoding = extract_vlad_encoding_tensor(image_tensor)

    return vlad_encoding

extracted_tensor = extract_vlad_encoding_path('checkboard_7.jpg', compress = True)
print(extracted_tensor)

import time

# Define image paths (update as needed)
image_path_1 = 'checkboard_7.jpg'  # Update with actual path
image_path_2 = 'checkboard_7.jpg'  # Update with actual path

# Start timing
start_time = time.time()

# Extract VLAD encodings for both images
vlad_encoding_1 = extract_vlad_encoding_path(image_path_1, compress=True, normalize=False).view(-1)  # Flatten the encoding
vlad_encoding_2 = extract_vlad_encoding_path(image_path_2, compress=True, normalize=False).view(-1)  # Flatten the encoding

# Calculate Euclidean distance
euclidean_distance = torch.dist(vlad_encoding_1, vlad_encoding_2, p=2)

# End timing
end_time = time.time()
execution_time = end_time - start_time

print(f"Euclidean distance between {image_path_1} and {image_path_2}: {euclidean_distance.item()}")
print(f"Execution time: {execution_time:.4f} seconds")