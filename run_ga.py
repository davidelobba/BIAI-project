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
from utils import load_config, get_classification_and_confidence, test_model, dataset_loader

from fitness import fitness
from toolbox_init import create_ga_toolbox

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
        directory = f'/home/disi/project2/ga_output/{config["network"]}'
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = f'/home/disi/project2/ga_output/{config["network"]}/gen_{g}_label_{label}_confidence_{confidence:.4f}.png'
        torchvision.utils.save_image(best_image, filename)

        wandb.log({
            "Best Label": label,
            "Best Confidence": confidence
        })

        print(f"Saved best image of generation {g} with label {label} and confidence {confidence:.4f}")
        print("-- End of (successful) evolution --")

if __name__ == "__main__":
    main()
