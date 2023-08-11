import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
from torchvision.models import resnet18, resnet34, resnet50, vit_b_16, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ViT_B_16_Weights
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


#class Network():
#    def __init__(self, network_name):
#        self.network_name = network_name
#        self.network = self.get_network()
#
#    def get_network(self):
#        network_dict = {
#            'resnet18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
#            'resnet34': resnet34(weights=ResNet34_Weights.IMAGENET1K_V1),
#            'resnet50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
#            'vit': vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#        }
#
#        if self.network_name in network_dict:
#            return self.create_network(network_dict[self.network_name])
#        else:
#            raise ValueError('Network not found')
#        
#    def create_network(self, network):
#        return network
    


class Network():
    def __init__(self, network_name, dataset='imagenet'):
        self.network_name = network_name
        self.dataset = dataset
        self.network = self.get_network()

    def get_network_weights(self, model_name):
        if self.dataset == 'cifar10':
            weights_path_dict = {
                'resnet18': '/home/disi/project/resnet_weights_cifar10/resnet18.pt',
                'resnet34': '/home/disi/project/resnet_weights_cifar10/resnet34.pt',
                'resnet50': '/home/disi/project/resnet_weights_cifar10/resnet50.pt',
                # Add other models here if needed
            }
            path = weights_path_dict[model_name]
            return torch.load(path)
        else:
            return None

    def get_network(self):
        network_dict = {
            'resnet18': resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if self.dataset == 'imagenet' else None),
            'resnet34': resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if self.dataset == 'imagenet' else None),
            'resnet50': resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if self.dataset == 'imagenet' else None),
            'vit': vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if self.dataset == 'imagenet' else None)
        }

        if self.network_name in network_dict:
            model = network_dict[self.network_name]
            if self.dataset == 'cifar10':
                model.load_state_dict(self.get_network_weights(self.network_name))
            return self.create_network(model)
        else:
            raise ValueError('Network not found')

    def create_network(self, network):
        return network