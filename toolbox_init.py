from deap import base, creator, tools, cma
import random


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

def create_pso_toolbox(fitness):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, 0, 1)
    toolbox.register("particle", tools.initRepeat, creator.Particle, toolbox.attr_float, n=3*224*224)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)

    toolbox.register("evaluate", fitness)
    toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)

    return toolbox

def updateParticle(part, best, phi1, phi2):
    u1 = (random.uniform(0, phi1) for _ in part)
    u2 = (random.uniform(0, phi2) for _ in part)
    v_u1 = map(lambda x: x[0] * x[1], zip(u1, part.best - part))
    v_u2 = map(lambda x: x[0] * x[1], zip(u2, best - part))
    part.speed = list(map(lambda x: x[0] + x[1], zip(v_u1, v_u2)))
    for i, speed in enumerate(part.speed):
        if abs(speed) < part.smin:
            part.speed[i] = part.smin
        elif abs(speed) > part.smax:
            part.speed[i] = part.smax
    part[:] = list(map(lambda x: x[0] + x[1], zip(part, part.speed)))
