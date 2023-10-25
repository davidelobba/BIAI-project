import numpy as np
import torch
import torch.nn.functional as F
from cppn_init import load_weights_into_cppn, generate_image
from utils import upsample_numpy_image
from PIL import Image


def fitness_ga(individual, model, dataset, transform=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dataset == 'mnist':
        image = torch.reshape(torch.tensor(individual), (1, 1, 224, 224)).to(device)
        image = transform['mnist'](image)
    else:
        image = torch.reshape(torch.tensor(individual), (1, 3, 224, 224)).to(device)
        image = transform['default'](image)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

def fitness_cma_es(individual, model, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if dataset == 'mnist':
        individual_upsampled = upsample_numpy_image(np.array(individual).reshape((1, 32, 32)), dataset)
    else:
        individual_upsampled = upsample_numpy_image(np.array(individual).reshape((3, 32, 32)), dataset)

    image = torch.tensor(individual_upsampled).float().unsqueeze(0).to(device)
    if dataset == 'mnist':
        image = image.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image).to(device)
        _, predicted = torch.max(outputs.data, 1)
        confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

def fitness_cppn(individual, cppn, model, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cppn = cppn.to(device)
    model = model.to(device)

    load_weights_into_cppn(cppn, individual)
    image = generate_image(cppn, dataset).to(device)

    with torch.no_grad():
        if dataset == 'mnist':
            image = torch.reshape(image, (1, 1, 224, 224))
        else:
            image = torch.reshape(image, (1, 3, 224, 224))

        outputs = model(image).to(device)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

def fitness_pso(particle, model, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if dataset == 'mnist':
        image = torch.reshape(torch.tensor(particle), (1, 1, 224, 224)).to(device)
    else:
        image = torch.reshape(torch.tensor(particle), (1, 3, 224, 224)).to(device)

    with torch.no_grad():
        outputs = model(image).to(device)
        _, predicted = torch.max(outputs.data, 1)
        confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

def fitness_prova(individual, model, target_class=None, temperature = 1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(np.array(individual).reshape((3, 224, 224))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

        # Applying temperature scaling to the logits before softmax
        outputs = outputs / temperature

        confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    # Targeted attack: reward if predicted is target_class and confidence is high
    if target_class is not None:
        if predicted.item() == target_class:
            return confidence.item()
        else:
            return 0.0

    # Untargeted attack: penalize high confidence for arbitrary "true" class (e.g., 0)
    # and reward high confidence for any other class
    else:
        if predicted.item() == 0:
            return 1.0 - confidence.item()
        else:
            return confidence.item()
