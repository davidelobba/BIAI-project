import os
import random
import numpy as np
import torch
from tqdm import tqdm
import yaml
import torchvision
from deap import tools
from torchvision import datasets, models, transforms
import torch.nn as nn

from networks import NetworkLoader

from utils import load_config, get_classification_and_confidence, test_model, dataset_loader
from fitness import fitness_pso as fitness
from toolbox_init import create_pso_toolbox

import wandb


def run_pso(args, weights_path):
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
        wandb.init(project="BIAI_project", config={"algorithm": "PSO", "dataset": args.dataset}, name = config['network'])

    toolbox = create_pso_toolbox(fitness, args.dataset)

    POP_SIZE, W, C1, C2, NGEN = config['pso']['population_size'], config['pso']['inertia'], config['pso']['cognitive_coefficient'], config['pso']['social_coefficient'], config['pso']['generations']

    particles = toolbox.population(n=POP_SIZE)

    for particle in particles:
        particle.speed = [random.uniform(0, 1) for _ in range(len(particle))]
        particle.pbest = toolbox.clone(particle)
    gbest = None

    print("Starting evolution")

    for g in range(NGEN):
        for particle in tqdm(particles, f"Evaluating generation {g}", leave=False):
            for i in range(len(particle)):
                inertia = W * particle.speed[i]
                cognitive = C1 * random.random() * (particle.pbest[i] - particle[i])
                social = 0

                if gbest is not None:
                    social = C2 * random.random() * (gbest[i] - particle[i])
                
                particle.speed[i] = inertia + cognitive + social
                particle[i] += particle.speed[i]

            particle.fitness.values = toolbox.evaluate(particle, network, args.dataset)

            if particle.fitness > particle.pbest.fitness:
                particle.pbest = toolbox.clone(particle)

        gbest = max(particles, key=lambda x: x.fitness.values[0])

        fits = [ind.fitness.values[0] for ind in particles]
        mean = sum(fits) / POP_SIZE
        std = abs(sum(x**2 for x in fits) / POP_SIZE - mean**2)**0.5

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

        label, confidence = get_classification_and_confidence(gbest, network, args.dataset)

        if args.dataset == 'mnist':
            best_image = torch.reshape(torch.tensor(gbest), (1, 1, 224, 224))
        else:
            best_image = torch.reshape(torch.tensor(gbest), (1, 3, 224, 224))

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
    