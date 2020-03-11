import numpy as np
from PIL import Image

import torchvision
import torch

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

cifar10_mean = (0.4913, 0.4821, 0.4465)
cifar10_std = (0.2470, 0.2434, 0.2615)

cifar10_train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, 
                             std=cifar10_std)
    ]
)

cifar10_test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, 
                             std=cifar10_std)
    ]
)

cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

cifar100_train_transforms = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, 
                             std=cifar100_std)
    ]
)

cifar100_test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, 
                             std=cifar100_std)
    ]
)
    

def load_cifar10(root='data/', batch_size=128, download=True, num_workers=0, drop_last=False):
    cifar10_train = CIFAR10(root, train=True, transform=cifar10_train_transforms, download=download)
    cifar10_test = CIFAR10(root, train=False, transform=cifar10_test_transforms, download=download)
    
    cifar10_train_dataloader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    cifar10_test_dataloader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return cifar10_train_dataloader, cifar10_test_dataloader


def load_cifar100(root='data/', batch_size=128, download=True, num_workers=0, drop_last=False):
    cifar100_train = CIFAR100(root, train=True, transform=cifar100_train_transforms, download=download)
    cifar100_test = CIFAR100(root, train=False, transform=cifar100_test_transforms, download=download)
    
    cifar100_train_dataloader = DataLoader(cifar100_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last)
    cifar100_test_dataloader = DataLoader(cifar100_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return cifar100_train_dataloader, cifar100_test_dataloader

