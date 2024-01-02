import numpy as np
import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0)


class CPPN(nn.Module):
    '''
    Code adapted from https://github.com/rystylee/pytorch-cppn-gan
    '''
    def __init__(self, dim_z, dim_c, ch):
        super(CPPN, self).__init__()
        dim_z = dim_z
        dim_c = dim_c
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


def get_coordinates(dim_x, dim_y, scale=1.0, batch_size=1):
    '''
    calculates and returns a vector of x and y coordinates, and corresponding radius from the centre of image.
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


def load_weights_into_cppn(cppn, individual):
    with torch.no_grad():
        idx = 0
        for param in cppn.parameters():
            num_params = param.numel()
            param.copy_(torch.tensor(individual[idx:idx+num_params]).view_as(param))
            idx += num_params
    return cppn