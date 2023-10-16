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


def run_cma_es(args, weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir

    config = load_config(args.config_path)
    loader = NetworkLoader(args)
    network = loader.load_network(weights_path, device)

    if args.test:
        dataloader = dataset_loader()
        criterion = nn.CrossEntropyLoss()
        test_model(network, criterion, dataloader['val'], device)

    if args.wandb:
        wandb.init(project="BIAI_project", config={"algorithm": "CMA-ES", "dataset": args.dataset}, name = config['network'])

    toolbox = create_cma_es_toolbox(fitness, args.dataset, sigma=config['cma_es']['sigma'], population_size=config['cma_es']['population_size'])

    population = toolbox.generate()

    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    for ind in tqdm(invalid_ind, desc="Evaluating initial population", leave=False):
        ind.fitness.values = toolbox.evaluate(ind, network, args.dataset)

    ngen = config['cma_es']['generations']

    for g in range(ngen):
        # Generate a new population
        offspring = toolbox.generate()
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        for ind in tqdm(invalid_ind, desc=f"Evaluating generation {g}", leave=False):
            ind.fitness.values = toolbox.evaluate(ind, network, args.dataset)

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

        if args.wandb:
            wandb.log({
                "Generation": g,
                "Min Fitness": min(fits),
                "Max Fitness": max(fits),
                "Average Fitness": mean,
                "Standard Deviation": std,
            })

        # Determine the best individual of the current generation
        best_ind = tools.selBest(offspring, 1)[0]

        if args.dataset == 'mnist':
            best_ind_upsampled = upsample_numpy_image(np.array(best_ind).reshape((1, 32, 32)), dataset=args.dataset)
        else:
            best_ind_upsampled = upsample_numpy_image(np.array(best_ind).reshape((3, 32, 32)), dataset=args.dataset)

        # Get the classification label and confidence
        label, confidence = get_classification_and_confidence(best_ind_upsampled, network, dataset=args.dataset)

        # Create the image from the best individual
        if args.dataset == 'mnist':
            best_image = torch.tensor(best_ind_upsampled.reshape((1, 224, 224))).float()
        else:
            best_image = torch.tensor(best_ind_upsampled.reshape((3, 224, 224))).float()

        if args.save:
            directory = os.path.join(output_dir, args.dataset, args.network)
            if not os.path.exists(directory):
                os.makedirs(directory)

            filename = directory + f'/gen_{g}_label_{label}_confidence_{confidence:.4f}.png'
            torchvision.utils.save_image(best_image, filename)

        if args.wandb:
            wandb.log({
                "Best Label": label,
                "Best Confidence": confidence
            })

        print(f"Best image of generation {g} has label {label} and confidence {confidence:.4f}")
    
    print("-- End of successful evolution --")