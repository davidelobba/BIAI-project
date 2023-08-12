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
from inference import evaluate_resnet18

from networks import Network
from fitness import fitness
from project2.toolbox_init import create_ga_toolbox

import wandb

def get_classification_and_confidence(individual, model):
    # Create the image from the individual
    image = torch.tensor(np.array(individual).reshape((3, 224, 224))).float().unsqueeze(0)

    with torch.no_grad():
        # Get the model's predictions
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        
        # Calculate the confidence of the prediction
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

    return predicted.item(), confidence.item()

def test_model(model, weights_path, criterion, dataloader, device):
    model.load_state_dict(torch.load(weights_path))
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloader):
        labels = torch.tensor(labels)

        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = running_corrects.double() / len(dataloader)

    print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #evaluate_resnet18("/home/disi/project2/weights/resnet18_best_model_wts_9467.pth")

    # Read config file
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # Load network
    #network = Network(config['network'])
    #network.network.eval()

    dataset = Dataset(config['dataset'])

    # Load the ResNet18 model and its weights
    weights_path = config['weights_path']

    network = models.resnet18(pretrained=True)
    num_ftrs = network.fc.in_features
    network.fc = nn.Linear(num_ftrs, 10)

    network.load_state_dict(torch.load(weights_path))
    network = network.to(device)

    test_model(network, weights_path, nn.CrossEntropyLoss(), dataset.dataset, device)
    
    #network = resnet34(pretrained=True)
    #network.eval()

    wandb.init(project="BIAI_project", config={"algorithm": "Genetic Algorithm"}, name = config['network'])

    # Create GA toolbox
    toolbox = create_ga_toolbox(fitness)

    # Evolutionary algorithm parameters
    POP_SIZE, CXPB, MUTPB, NGEN = config['ga']['population_size'], config['ga']['crossover_probability'], config['ga']['mutation_probability'], config['ga']['generations']
    
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
            #ind.fitness.values = toolbox.evaluate(ind, network.network)
            ind.fitness.values = toolbox.evaluate(ind, network)

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

        # Get the classification label and confidence
        #label, confidence = get_classification_and_confidence(best_ind, network.network)
        label, confidence = get_classification_and_confidence(best_ind, network)

        # Create the image from the best individual
        best_image = torch.tensor(np.array(best_ind).reshape((3, 224, 224))).float()

        # Save the image. Make sure to replace 'path/to/save' with your actual directory
        filename = f'/home/disi/project2/output/gen_{g}_label_{label}_confidence_{confidence:.4f}.png'
        torchvision.utils.save_image(best_image, filename)

        wandb.log({
            "Best Label": label,
            "Best Confidence": confidence
        })

        print(f"Saved best image of generation {g} with label {label} and confidence {confidence:.4f}")
        print("-- End of (successful) evolution --")

if __name__ == "__main__":
    main()

