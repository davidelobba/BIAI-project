import torch
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
import torch.nn as nn

from utils import load_config

'''def get_network_by_name(network_name):
    # Add or remove network models from the dictionary as needed
    networks = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50
    }
    return networks.get(network_name, None)()

class NetworkLoader:
    def __init__(self, args):
        self.args = args

    def load_network(self, weights_path, device):
        network_name = self.args.network

        network = get_network_by_name(network_name)
        if network is None:
            raise ValueError(f"Network {network_name} not recognized.")

        # If the network is ResNet-like, change the final layer. Modify this logic for other architectures.
        if "resnet" in network_name and self.args.dataset == 'cifar10':
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, 10)
            network.load_state_dict(torch.load(weights_path))
            network = network.to(device)
            network.eval()

        elif "resnet" in network_name and self.args.dataset == 'imagenet':
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, 1000)
            weights = ResNet18_Weights.DEFAULT if network_name == 'resnet18' else ResNet34_Weights.DEFAULT if network_name == 'resnet34' else ResNet50_Weights.DEFAULT
            network = network(weights).to(device)
            network.eval()
        

        return network'''
    


import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

class NetworkLoader():
    def __init__(self, args):
        self.network_name = args.network
        self.dataset = args.dataset

    def _initialize_network(self, pretrained=False):
        if self.network_name == 'resnet18':
            return resnet18(pretrained=pretrained)
        elif self.network_name == 'resnet34':
            return resnet34(pretrained=pretrained)
        elif self.network_name == 'resnet50':
            return resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Network {self.network_name} not recognized.")
    
    def _adjust_output_layer(self, network):
        num_ftrs = network.fc.in_features
        if self.dataset == 'cifar10':
            network.fc = nn.Linear(num_ftrs, 10)
        elif self.dataset == 'imagenet':
            network.fc = nn.Linear(num_ftrs, 1000)
        elif self.dataset == 'mnist':
            network.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            network.fc = nn.Linear(num_ftrs, 10)
        else:
            raise ValueError(f"Dataset {self.dataset} not supported.")

    def load_network(self, weights_path=None, device='cpu'):
        if self.dataset == 'imagenet':
            network = self._initialize_network(pretrained=True)
        else:
            network = self._initialize_network(pretrained=False)
            
        self._adjust_output_layer(network)
        
        if self.dataset == 'cifar10' and weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            network.load_state_dict(state_dict)
        if self.dataset == 'mnist' and weights_path:
            state_dict = torch.load(weights_path, map_location=device)
            network.load_state_dict(state_dict)

        print(f"Loaded {self.network_name} network.")
        
        network.to(device)
        network.eval()
        return network
