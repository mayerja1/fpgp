#!/usr/bin/env python3
import math
import numpy as np
import time
import matplotlib.pyplot as plt

from symb_reg_pset import pset
from symb_reg_toolbox import toolbox
import fitness_pred

from deap import gp
from deap import algorithms
from deap import tools
from deap import algorithms
from deap import creator

import copy


CXPB = 0.5
MUTPB = 0.1
POP_SIZE = 128

class SymbRegTree(gp.PrimitiveTree):

    def __init__(self, content):
        super().__init__(content)

    def eval_at_points(self, points, compile_func, out):
        func = compile_func(self)
        for i, p in enumerate(points):
            out[i] = func(p)

def target_func(x):
    return math.exp(abs(x))*math.sin(2 * math.pi * x)

def symbreg_fitness(individual_idx, points, population_values):
    assert(len(population_values[individual_idx]) == len(points))
    vals = (abs(v - target_func(p)) for v, p in zip(population_values[individual_idx], points))
    return math.fsum(vals) / len(points),

def symb_reg_with_fp(population, toolbox, cxpb, mutpb, end_cond, end_func, fp, points, stats=None,
                     halloffame=None, verbose=__debug__):

    def _terminate():
        vars = [gen, evals, time, best_fitness]
        vars_name_mapping = {'gen' : gen, 'evals' : evals, 'time' : time.time() - start_time, 'best_fitness' : best_fitness}
        return end_func(vars_name_mapping[end_cond])

    logbook = tools.Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])

    gen = 0
    evals = 0
    start_time = time.time()
    best_fitness = (np.inf,)


    # array of values at points of the population
    population_values = np.zeros((len(population), len(fp.get_best_predictor())))

    # Begin the generational process
    while not _terminate():
        gen += 1
        # evaluation phase

        # get points to use
        predictor = fp.get_best_predictor()

        # check population_values shape
        if population_values.shape[1] != len(predictor):
            population_values = np.zeros((len(population), len(predictor)))

        # evaluate individuals at used points
        evals += len(population) * len(predictor)
        for i, ind in enumerate(population):
            ind.eval_at_points(points[predictor], toolbox.compile, population_values[i])

        # now compute fitnesses
        fitnesses = map(lambda i: symbreg_fitness(i, points[predictor], population_values), range(len(population)))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, **record)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

        if verbose:
            print(logbook.stream)

        # selection phase
        offspring = toolbox.select(population, len(population))

        # mutation and crossover phase
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Replace the current population by the offspring
        population[:] = offspring

    # TODO: this population is before selection
    return population, logbook

def run():
    # initialization
    creator.create("Individual", SymbRegTree, fitness=creator.FitnessMin)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", symbreg_fitness)

    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=False)

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    # stats we want to keep track of
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    points = np.linspace(-3, 3, 200)

    pop, log = symb_reg_with_fp(pop, toolbox, CXPB, MUTPB, 'gen', lambda x: x > 200,
                                fitness_pred.ExactFitness(len(points)), points, stats=mstats,
                                halloffame=hof, verbose=False)

    print(hof[0])
    print(hof[0].fitness)
    print(log[-1])
    plt.scatter(points, [target_func(x) for x in points], marker='+')
    best_vals = np.zeros(len(points))
    hof[0].eval_at_points(points, toolbox.compile, best_vals)
    plt.plot(points, best_vals)
    #plt.show()


if __name__ == '__main__':
    run()
