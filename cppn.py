import os
import random
import numpy as np
import torch
from tqdm import tqdm
import yaml
import torch.nn as nn
from networks import NetworkLoader
from deap import tools
import torchvision
from cppn_init import CPPN, load_weights_into_cppn, generate_image
from utils import load_config, get_classification_and_confidence_cppn, test_model, dataset_loader

from fitness import fitness_cppn
from toolbox_init import create_cppn_toolbox
import wandb


def run_cppn(args, weights_path):

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
        wandb.init(project="BIAI_project", config={"algorithm": "CPPN"}, name = config['network'])

    cppn_model = CPPN()
    ind_size = sum(p.numel() for p in cppn_model.parameters())
    toolbox = create_cppn_toolbox(fitness_cppn, cppn_model, ind_size, network)

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

        if args.wandb:
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
        label, confidence = get_classification_and_confidence_cppn(best_image, network)

        if args.save:
            directory = os.path.join(output_dir, args.network)
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