#loading of the dataset
import torch
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Dataset():
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset = self.get_dataset()

    def get_dataset(self):
        dataset_dict = {
            'cifar10': datasets.CIFAR10,
            'cifar100': datasets.CIFAR100,
            'imagenet': datasets.ImageNet,
            'mnist': datasets.MNIST
        }

        if self.dataset_name in dataset_dict:
            return self.create_dataset(dataset_dict[self.dataset_name])
        else:
            print('Dataset not found')
            return None
    
    def get_subset(self, n=1000):
        return torch.utils.data.Subset(self.dataset, range(n))

    def create_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])

    def create_dataset(self, dataset_class):
        if self.dataset_name == 'imagenet':
            return dataset_class(root='./data/', split='val', download=False, transform=self.create_transform())
        else:
            return dataset_class(root='./data', train=False, download=True, transform=self.create_transform())
