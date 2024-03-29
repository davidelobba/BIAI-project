from deap import base, creator, tools, cma
import random
import numpy as np


def create_ga_toolbox(fitness, dataset):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)

    if dataset == 'mnist':
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1*224*224)
    else:
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3*224*224)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", clip_mutation, mu=0, sigma=0.1, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def create_cma_es_toolbox(fitness, dataset, sigma=0.5, population_size=10):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("evaluate", fitness)

    if dataset == 'mnist':
        strategy = cma.Strategy(centroid=[0.5]*(1*32*32), sigma=sigma, lambda_=population_size)
    else:
        strategy = cma.Strategy(centroid=[0.5]*(3*32*32), sigma=sigma, lambda_=population_size)

    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    return toolbox


def create_cppn_toolbox(fitness, ind_size):    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", fitness)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def clip_mutation(individual, *args, **kwargs):
    tools.mutGaussian(individual, *args, **kwargs)
    for i in range(len(individual)):
        individual[i] = min(max(0, individual[i]), 1)
    return individual,
    
    
