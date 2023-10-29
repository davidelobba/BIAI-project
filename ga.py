import os
import random
import numpy as np
import torch
from tqdm import tqdm
import torchvision
from deap import tools
import torch.nn as nn

from networks import NetworkLoader
from utils import load_config, get_classification_and_confidence, test_model, dataset_loader, get_transform

from fitness import fitness_ga as fitness
from toolbox_init import create_ga_toolbox

import wandb

def run_ga(args, weights_path):
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
        dataloader = dataset_loader(args.dataset)
        criterion = nn.CrossEntropyLoss()
        test_model(network, criterion, dataloader['val'], device)

    if args.wandb:
        wandb.init(project="BIAI_project", config=vars(args), name=f"{args.algorithm}_{args.network}_{args.dataset}")

    toolbox = create_ga_toolbox(fitness, args.dataset)

    POP_SIZE, CXPB, MUTPB, NGEN = config['ga']['population_size'], config['ga']['crossover_probability'], config['ga']['mutation_probability'], config['ga']['generations']

    if args.adaptive_mutation_crossover:
        STAGNATION_LIMIT = config['ga']['stagnation']
        MAX_MUTPB, MIN_MUTPB, MUTPB_DELTA = config['ga']['max_mutpb'], config['ga']['min_mutpb'], config['ga']['mutpb_delta']
        MAX_CXPB, MIN_CXPB, CXPB_DELTA = config['ga']['max_cxpb'], config['ga']['min_cxpb'], config['ga']['cxpb_delta']
        best_fitnesses = [float('-inf')] * STAGNATION_LIMIT
  
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
            ind.fitness.values = toolbox.evaluate(ind, network, args.dataset, transform)

        print("Evaluated %i individuals" % len(invalid_ind))

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        avg_fit = sum(fits) / len(fits)

        if args.adaptive_mutation_crossover:
            best_fitness_current_gen = max(fits)

            generation_factor = g / NGEN
            MUTPB = MAX_MUTPB - (MAX_MUTPB - MIN_MUTPB) * generation_factor
            CXPB = MIN_CXPB + (MAX_CXPB - MIN_CXPB) * generation_factor

            if best_fitness_current_gen <= min(best_fitnesses):
                MUTPB = min(MAX_MUTPB, MUTPB + MUTPB_DELTA)
                CXPB = max(MIN_CXPB, CXPB - CXPB_DELTA)
            else:
                MUTPB = max(MIN_MUTPB, MUTPB - MUTPB_DELTA)
                CXPB = min(MAX_CXPB, CXPB + CXPB_DELTA)

            best_fitnesses.pop(0)
            best_fitnesses.append(best_fitness_current_gen)
            print(f"Adaptive mutation and crossover probabilities: {MUTPB}, {CXPB}")

        length = len(pop)
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

        # Determine the best individual of the current generation
        best_ind = tools.selBest(pop, 1)[0]

        # Get the classification label and confidence
        label, confidence = get_classification_and_confidence(best_ind, network, args.dataset, transform)

        if args.dataset == 'mnist':
            best_image = torch.tensor(np.array(best_ind).reshape((1, 224, 224))).float()
        else:
            best_image = torch.tensor(np.array(best_ind).reshape((3, 224, 224))).float()

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
