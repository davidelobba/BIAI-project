import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class NetworkLoader():
    def __init__(self, args):
        self.network_name = args.network
        self.dataset = args.dataset

    def _initialize_network(self):
        if self.network_name == 'resnet18' and self.dataset == 'imagenet':
            return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif self.network_name == 'resnet34' and self.dataset == 'imagenet':
            return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        elif self.network_name == 'resnet50' and self.dataset == 'imagenet':
            return resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif self.network_name == 'resnet18':
            return resnet18(weights=None)
        elif self.network_name == 'resnet34':
            return resnet34(weights=None)
        elif self.network_name == 'resnet50':
            return resnet50(weights=None)
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
        network = self._initialize_network()    
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
