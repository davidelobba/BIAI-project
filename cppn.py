import torch.nn as nn
import torch


class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
    
class CPPN(nn.Module):
    def __init__(self):
        super(CPPN, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(2, 20),
            Sine(),
            nn.Linear(20, 20),
            Sine(),
            nn.Linear(20, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)


def load_weights_into_cppn(cppn, individual):
    with torch.no_grad():
        idx = 0
        for param in cppn.parameters():
            num_params = param.numel()
            param.copy_(torch.tensor(individual[idx:idx+num_params]).view_as(param))
            idx += num_params

def generate_image(cppn, width=224, height=224):
    image = torch.zeros(3, width, height)
    for x in range(width):
        for y in range(height):
            coord = torch.tensor([x / width, y / height]).float()
            image[:, x, y] = cppn(coord)
    return image