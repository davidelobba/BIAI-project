import numpy as np
import torch
import torch.nn.functional as F
from cppn import load_weights_into_cppn, generate_image

from PIL import Image

def upsample_numpy_image(image_np, target_size=(224, 224)):
    """Upsample the given numpy image array to the target size using bilinear interpolation."""
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8).transpose(1, 2, 0))
    image_pil_upsampled = image_pil.resize(target_size, Image.BILINEAR)
    return np.array(image_pil_upsampled).transpose(2, 0, 1) / 255.0

def get_classification_and_confidence(image, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.float().unsqueeze(0).to(device)  # Ensure the image is on the correct device

    with torch.no_grad():
        # Get the model's predictions
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate the confidence of the prediction
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

    return predicted.item(), confidence.item()

def fitness(individual, model):
    # Create the image from the individual
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(np.array(individual).reshape((3, 224, 224))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # Get the model's predictions
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate the confidence of the prediction
        confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    # Fitness is the confidence of the prediction
    return (confidence,)

def fitness_cma_es(individual, model):
    # Create the image from the individual
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    individual_upsampled = upsample_numpy_image(np.array(individual).reshape((3, 32, 32)))

    image = torch.tensor(individual_upsampled).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # Get the model's predictions
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate the confidence of the prediction
        confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    # Fitness is the confidence of the prediction
    return (confidence,)

def fitness_cppn(individual, cppn, target_network):
    # Load the weights from the individual into the CPPN
    load_weights_into_cppn(cppn, individual)
    
    # Generate the image using the CPPN
    generated_image = generate_image(cppn)

    _, confidence = get_classification_and_confidence(generated_image, target_network)
    
    return confidence,

def fitness_pso(particle, model):
    # Create the image from the individual
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = torch.tensor(np.array(particle).reshape((3, 224, 224))).float().unsqueeze(0).to(device)

    with torch.no_grad():
        # Get the model's predictions
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate the confidence of the prediction
        confidence = F.softmax(outputs, dim=1)[0][predicted.item()]

    # Fitness is the confidence of the prediction
    return (confidence,)