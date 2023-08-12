import os
import random
import numpy as np
import torch
from tqdm import tqdm
import yaml
import torchvision
from deap import tools
import torch.nn as nn

from networks import NetworkLoader
from utils import load_config, get_classification_and_confidence, test_model, dataset_loader, upsample_numpy_image

from fitness import fitness_cma_es as fitness
from toolbox_init import create_cma_es_toolbox

import wandb


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = load_config("config.yaml")

    dataloader = dataset_loader()

    weights_path = config['weights_path']

    loader = NetworkLoader()
    network = loader.load_network(weights_path, device)


    criterion = nn.CrossEntropyLoss()

    test_model(network, criterion, dataloader['val'], device)

    wandb.init(project="BIAI_project", config={"algorithm": "CMA-ES"}, name = config['network'])

    # Create GA toolbox
    toolbox = create_cma_es_toolbox(fitness, ngen=config['cma_es']['generations'], sigma=config['cma_es']['sigma'], population_size=config['cma_es']['population_size'])

    population = toolbox.generate()

    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    for ind in tqdm(invalid_ind, desc="Evaluating initial population", leave=False):
        ind.fitness.values = toolbox.evaluate(ind, network)

    ngen = config['cma_es']['generations']

    for g in range(ngen):
        # Generate a new population
        offspring = toolbox.generate()
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        for ind in tqdm(invalid_ind, desc=f"Evaluating generation {g}", leave=False):
            ind.fitness.values = toolbox.evaluate(ind, network)

        # Update the strategy with the evaluated individuals
        toolbox.update(offspring)

        print("Evaluated %i individuals" % len(invalid_ind))

        fits = [ind.fitness.values[0] for ind in offspring]

        length = len(offspring)
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
        best_ind = tools.selBest(offspring, 1)[0]

        best_ind_upsampled = upsample_numpy_image(np.array(best_ind).reshape((3, 32, 32)))

        # Get the classification label and confidence
        #label, confidence = get_classification_and_confidence(best_ind, network.network)
        label, confidence = get_classification_and_confidence(best_ind_upsampled, network)

        # Create the image from the best individual
        best_image = torch.tensor(best_ind_upsampled.reshape((3, 224, 224))).float()

        directory = f'/home/disi/project2/cma_es_output/{config["network"]}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the image. Make sure to replace 'path/to/save' with your actual directory
        filename = f'/home/disi/project2/cma_es_output/{config["network"]}/gen_{g}_label_{label}_confidence_{confidence:.4f}.png'
        torchvision.utils.save_image(best_image, filename)

        wandb.log({
            "Best Label": label,
            "Best Confidence": confidence
        })

        print(f"Saved best image of generation {g} with label {label} and confidence {confidence:.4f}")
        print("-- End of (successful) evolution --")

if __name__ == "__main__":
    main()
