import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

'''class CPPN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=32, output_dim=3):  # output_dim is 3 for RGB
        super(CPPN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.sin(self.fc1(x))
        x = torch.sin(self.fc2(x))
        x = F.sigmoid(self.fc3(x))  # normalize output to [0, 1]

        return x'''

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

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)

'''
class CPPN(nn.Module):
    def __init__(self, z, c, ch):
        super(CPPN, self).__init__()
        dim_z = z
        dim_c = c
        ch = ch

        self.l_z = nn.Linear(dim_z, ch)
        self.l_x = nn.Linear(1, ch, bias=False)
        self.l_y = nn.Linear(1, ch, bias=False)
        self.l_r = nn.Linear(1, ch, bias=False)

        self.ln_seq = nn.Sequential(
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),

            nn.Linear(ch, ch),
            nn.Tanh(),

            nn.Linear(ch, dim_c),
            nn.Sigmoid()
            )

        self._initialize()

    def _initialize(self):
        self.apply(weights_init)

    def forward(self, z, x, y, r):
        u = self.l_z(z) + self.l_x(x) + self.l_y(y) + self.l_r(r)
        out = self.ln_seq(u)
        return out
'''


def get_coordinates(dim_x, dim_y, scale=1.0, batch_size=1):
    '''
    calculates and returns a vector of x and y coordintes, and corresponding radius from the centre of image.
    '''
    n_points = dim_x * dim_y
    x_range = scale * (np.arange(dim_x) - (dim_x - 1) / 2.0) / (dim_x - 1) / 0.5
    y_range = scale * (np.arange(dim_y) - (dim_y - 1) / 2.0) / (dim_y - 1) / 0.5
    x_mat = np.matmul(np.ones((dim_y, 1)), x_range.reshape((1, dim_x)))
    y_mat = np.matmul(y_range.reshape((dim_y, 1)), np.ones((1, dim_x)))
    r_mat = np.sqrt(x_mat * x_mat + y_mat * y_mat)
    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, n_points, 1)
    return torch.from_numpy(x_mat).float(), torch.from_numpy(y_mat).float(), torch.from_numpy(r_mat).float()
