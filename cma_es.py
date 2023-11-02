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
from utils import load_config, get_classification_and_confidence, test_model, dataset_loader, upsample_numpy_image, get_transform

from fitness import fitness_cma_es as fitness
from toolbox_init import create_cma_es_toolbox

import wandb


def run_cma_es(args, weights_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir

    if args.normalize:
        transform = get_transform()
    else:
        transform = None

    config = load_config(args.config_path)
    loader = NetworkLoader(args)
    network = loader.load_network(weights_path, device)

    if args.test:
        dataloader = dataset_loader()
        criterion = nn.CrossEntropyLoss()
        test_model(network, criterion, dataloader['val'], device)

    if args.wandb:
        wandb.init(project="BIAI_project", config=vars(args), name=f"{args.algorithm}_{args.network}_{args.dataset}")

    SIGMA, POP_SIZE, NGEN = config['cma_es']['sigma'], config['cma_es']['population_size'], config['cma_es']['generations']
    toolbox = create_cma_es_toolbox(fitness, args.dataset, sigma=SIGMA, population_size=POP_SIZE)

    pop = toolbox.generate()

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]

    for ind in tqdm(invalid_ind, desc="Evaluating initial population", leave=False):
        ind.fitness.values = toolbox.evaluate(ind, network, args.dataset, transform, args.class_constraint)

    for g in range(NGEN):
        offspring = toolbox.generate()
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        for ind in tqdm(invalid_ind, desc=f"Evaluating generation {g}", leave=False):
            ind.fitness.values = toolbox.evaluate(ind, network, args.dataset, transform, args.class_constraint)

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
                "Pop size": POP_SIZE,
                "Min Fitness": min(fits),
                "Max Fitness": max(fits),
                "Average Fitness": mean,
                "Standard Deviation": std,
            })

        best_ind = tools.selBest(offspring, 1)[0]

        if args.dataset == 'mnist':
            best_ind_upsampled = upsample_numpy_image(np.array(best_ind).reshape((1, 32, 32)), dataset=args.dataset)
        else:
            best_ind_upsampled = upsample_numpy_image(np.array(best_ind).reshape((3, 32, 32)), dataset=args.dataset)

        label, confidence = get_classification_and_confidence(best_ind_upsampled, network, dataset=args.dataset, transform=transform, target_class=args.class_constraint)

        if args.dataset == 'mnist':
            best_image = torch.tensor(best_ind_upsampled.reshape((1, 224, 224))).float()
        else:
            best_image = torch.tensor(best_ind_upsampled.reshape((3, 224, 224))).float()

        if args.save:
            directory = os.path.join(output_dir, args.dataset, args.network)
            if args.class_constraint is not None:
                directory = directory + f'/{args.class_constraint}'
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