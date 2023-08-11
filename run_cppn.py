import torch
import torchvision

from cppn import CPPN
from cppn import get_coordinates

import random
import numpy as np
import torch
from tqdm import tqdm
import yaml
import torchvision
from deap import tools
from torchvision import datasets, models, transforms
import torch.nn as nn

from dataset import Dataset

from cppn import CPPN


from networks import Network
from fitness import cppn_fitness
from ga_init import create_cppn_toolbox

import wandb

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


def test_model(model, criterion, dataloader, device):
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)


    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

def dataset_loader():
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=data_transforms['val'])
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    dataloader = {'train': trainloader, 'val': testloader}
    
    return dataloader

def network_loader(network_name, weights_path, device):
    network = models.resnet18(pretrained=False)
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, 10)
    network.load_state_dict(torch.load(weights_path))
    network = network.to(device)
    network.eval()

    return network

def load_config(file_path="config.yaml"):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def get_network_by_name(network_name):
    # Add or remove network models from the dictionary as needed
    networks = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50
    }
    return networks.get(network_name, None)()

class NetworkLoader:
    def __init__(self, config_file="config.yaml"):
        self.config = load_config(config_file)

    def load_network(self, weights_path, device):
        network_name = self.config['network']

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

def evaluate_cppn(individual, cppn, target_network):
    # Load the weights from the individual into the CPPN
    load_weights_into_cppn(cppn, individual)
    
    # Generate the image using the CPPN
    generated_image = generate_image(cppn)

    label, confidence = get_classification_and_confidence(generated_image, target_network)
    
    true_label = 0
    fitness_value = confidence if label != true_label else 0
    
    return fitness_value,


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = load_config("config.yaml")

    dataloader = dataset_loader()

    weights_path = config['weights_path']

    loader = NetworkLoader()
    network = loader.load_network(weights_path, device)


    criterion = nn.CrossEntropyLoss()

    #test_model(network, criterion, dataloader['val'], device)

    wandb.init(project="BIAI_project", config={"algorithm": "CPPN"}, name = config['network'])

    cppn_model = CPPN()
    ind_size = sum(p.numel() for p in cppn_model.parameters())
    toolbox = create_cppn_toolbox(evaluate_cppn, cppn_model, ind_size, network)

    POP_SIZE, CXPB, MUTPB, NGEN = config['cppn']['population_size'], config['cppn']['crossover_probability'], config['cppn']['mutation_probability'], config['cppn']['generations']
    
    pop = toolbox.population(n=POP_SIZE)

    print("Starting evolution")

    for g in range(NGEN):

        print(f"-- Generation {g} --")

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        for ind in tqdm(invalid_ind, desc="Evaluating individuals", leave=False):
            ind.fitness.values = toolbox.evaluate(ind)


        print("Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

        wandb.log({
            "Generation": g,
            "Min Fitness": min(fits),
            "Max Fitness": max(fits),
            "Average Fitness": mean,
            "Standard Deviation": std,
        })

        # Determine the best individual of the current generation
        best_ind = tools.selBest(pop, 1)[0]

        # Load the weights from the best individual into the CPPN
        load_weights_into_cppn(cppn_model, best_ind)

        # Generate the image using the CPPN
        best_image = generate_image(cppn_model)

        # Get the classification label and confidence for the generated image
        label, confidence = get_classification_and_confidence(best_image, network)

        # Save the image. Make sure to replace 'path/to/save' with your actual directory
        filename = f'/home/disi/project2/cppn_output/{config["network"]}/gen_{g}_label_{label}_confidence_{confidence:.4f}.png'
        torchvision.utils.save_image(best_image, filename)

        wandb.log({
            "Best Label": label,
            "Best Confidence": confidence
        })

        print(f"Saved best image of generation {g} with label {label} and confidence {confidence:.4f}")
        print("-- End of (successful) evolution --")


if __name__ == "__main__":
    main()
