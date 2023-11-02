import numpy as np
import torch
import torch.nn.functional as F
from utils import upsample_numpy_image
from cppn_init import load_weights_into_cppn


def fitness_ga(individual, model, dataset, transform=None, target_class=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset == 'mnist':
        image = torch.reshape(torch.tensor(individual), (1, 1, 224, 224)).to(device)
    else:
        image = torch.reshape(torch.tensor(individual), (1, 3, 224, 224)).to(device)

    if transform is not None:
        image = transform[dataset](image)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        if target_class is not None:
            confidence = F.softmax(outputs, dim=1)[0][target_class]
        else:
            confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

def fitness_cma_es(individual, model, dataset, transform=None, target_class=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    if dataset == 'mnist':
        image = upsample_numpy_image(np.array(individual).reshape((1, 32, 32)), dataset)
    else:
        image = upsample_numpy_image(np.array(individual).reshape((3, 32, 32)), dataset)

    image = torch.tensor(image).float().unsqueeze(0).to(device)
    if dataset == 'mnist':
        image = image.unsqueeze(0)

    if transform is not None:
        image = transform[dataset](image)

    with torch.no_grad():
        outputs = model(image).to(device)
        _, predicted = torch.max(outputs, 1)
        if target_class is not None:
            confidence = F.softmax(outputs, dim=1)[0][target_class]
        else:
            confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

def fitness_cppn(individual, cppn, model, dataset, z_scaled, x, y, r, transform=None, target_class=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cppn = load_weights_into_cppn(cppn, individual)

    with torch.no_grad():
        if dataset == 'mnist':
            image = cppn(z_scaled, x, y, r)
            image = image.view(-1, 224, 224, 1)
            image = image.permute((0, 3, 1, 2))
        else:
            image = cppn(z_scaled, x, y, r)
            image = image.view(-1, 224, 224, 3)
            image = image.permute((0, 3, 1, 2))

        if transform is not None:
            image = transform[dataset](image)

        outputs = model(image).to(device)
        _, predicted = torch.max(outputs, 1)
        if target_class is not None:
            confidence = F.softmax(outputs, dim=1)[0][target_class]
        else:
            confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

def fitness_pso(particle, model, dataset, transform=None, target_class=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if dataset == 'mnist':
        image = torch.reshape(torch.tensor(particle), (1, 1, 224, 224)).to(device)
    else:
        image = torch.reshape(torch.tensor(particle), (1, 3, 224, 224)).to(device)

    if transform is not None:
        image = transform[dataset](image)

    with torch.no_grad():
        outputs = model(image).to(device)
        _, predicted = torch.max(outputs, 1)
        if target_class is not None:
            confidence = F.softmax(outputs, dim=1)[0][target_class]
        else:
            confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)
