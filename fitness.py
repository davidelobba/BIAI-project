import numpy as np
import torch
import torch.nn.functional as F
from cppn_init import load_weights_into_cppn, generate_image
from utils import upsample_numpy_image
from PIL import Image



def get_classification_and_confidence(image, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.float().unsqueeze(0).to(device)
    model = model.to(device)

    with torch.no_grad():
        outputs = model(image).to(device)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

    return predicted.item(), confidence.item()

def fitness(individual, model, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if dataset == 'mnist':
        image = torch.tensor(np.array(individual).reshape((1, 224, 224))).float().unsqueeze(0).to(device)
    else:
        image = torch.tensor(np.array(individual).reshape((3, 224, 224))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image).to(device)
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
            image_np = image.cpu().numpy().reshape((1, 224, 224))
            image = torch.tensor(image_np).float().unsqueeze(0).to(device)
        else:    
            image_np = image.cpu().numpy().reshape((3, 224, 224))
            image = torch.tensor(image_np).float().unsqueeze(0).to(device)
        outputs = model(image).to(device)
        _, predicted = torch.max(outputs.data, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

def fitness_pso(particle, model, dataset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if dataset == 'mnist':
        image = torch.tensor(np.array(particle).reshape((1, 224, 224))).float().unsqueeze(0).to(device)
    else:            
        image = torch.tensor(np.array(particle).reshape((3, 224, 224))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image).to(device)
        _, predicted = torch.max(outputs.data, 1)
        confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    return (confidence,)

################################## Sistemare CPPN BOTTLENECK ##################################