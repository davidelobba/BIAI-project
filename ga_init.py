from deap import base, creator, tools, cma
import numpy as np
import random
from fitness import cppn_fitness
import torch

def create_ga_toolbox(fitness):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3*224*224)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def create_cma_es_toolbox(fitness, ngen=1000, sigma=0.5, population_size=10):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", fitness)

    # Create a strategy for CMA-ES
    strategy = cma.Strategy(centroid=[0.5]*(3*32*32), sigma=sigma, lambda_=population_size)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    return toolbox


def create_cppn_toolbox(fitness, cppn_model, ind_size, network):
    """
    Create a DEAP toolbox for evolving CPPNs.
    
    Parameters:
    - fitness_function: Function to evaluate the fitness of an individual.
    - cppn_model: The architecture of the CPPN.
    - ind_size: The number of weights in the CPPN model.
    
    Returns:
    - Initialized toolbox for genetic algorithm.
    """
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)  # Initialize CPPN weights between [-1, 1]
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness, cppn=cppn_model, target_network=network)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox