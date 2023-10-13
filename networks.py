import torch
from torchvision import models
from torchvision.models import resnet18, resnet34, resnet50, vit_b_16, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ViT_B_16_Weights
import torch.nn as nn

from utils import load_config

def get_network_by_name(network_name):
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
        if "resnet" in network_name:
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, 10)

        network.load_state_dict(torch.load(weights_path))
        network = network.to(device)
        network.eval()

        return network