#!/usr/bin/env python3
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import operator
import traceback
import copy

from symb_reg_pset import pset
from symb_reg_toolbox import toolbox
import fitness_pred

from deap import gp
from deap import algorithms
from deap import tools
from deap import algorithms
from deap import creator


class SymbRegTree(gp.PrimitiveTree):

    '''
    list of atributes that aren't copied during deepcopy
    it is a significant important speedup of the program
    there is no reason to copy these attributes because
    for individuals with invalid fitness they are computed again
    '''
    deepcopy_ignorelist = ('values', 'error_vec')

    def __init__(self, content):
        super().__init__(content)
        self.values = []
        self.error_vec = None

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        for k, v in self.__dict__.items():
            if k in SymbRegTree.deepcopy_ignorelist:
                setattr(new, k, v)
            else:
                setattr(new, k, copy.deepcopy(v))
        return new

    def eval_at_points(self, points, compile_func):
        func = compile_func(self)
        if len(self.values) != len(points):
            self.values = [0] * len(points)
        for i, p in enumerate(points):
            self.values[i] = func(p)

    def set_error_vec(self, target_values):
        assert(len(self.values) == len(target_values))
        self.error_vec = target_values - self.values


def deterministic_crowding(population, points, toolbox, cxpb, mutpb):
    assert(len(population) % 2 == 0)
    parents = [toolbox.clone(ind) for ind in population]
    offspring = []
    random.shuffle(parents)
    tree_values = np.zeros((4, len(points)))
    error_vectors = np.zeros_like(tree_values)
    nevals = 0
    for i in range(1, len(parents), 2):
        p1, p2 = toolbox.map(toolbox.clone, (parents[i - 1], parents[i]))
        # crossover
        c1, c2 = toolbox.mate(p1, p2)
        # make crossovered individuals' fitnesses invalid
        del c1.fitness.values, c2.fitness.values
        # p1 and p2 are crossed over, we need to get back to their originals
        p1, p2 = parents[i - 1], parents[i]

        # if we weren't supposed to crossover, use the original parent
        if random.random() > cxpb: c1 = toolbox.clone(p1)
        if random.random() > cxpb: c2 = toolbox.clone(p2)

        # mutate
        if random.random() < mutpb:
            c1, = toolbox.mutate(c1)
            del c1.fitness.values
        if random.random() < mutpb:
            c2, = toolbox.mutate(c2)
            del c2.fitness.values
        family = (p1, p2, c1, c2)
        invalid_ind = (ind for ind in family if not ind.fitness.valid)
        family_idx = {'p1' : 0, 'p2' : 1, 'c1' : 2, 'c2' : 3}
        # selection
        # evaluate, get errors and fitnesses
        target_values = np.array([toolbox.target_func(x) for x in points])
        for ind in invalid_ind:
            ind.eval_at_points(points, toolbox.compile)
            nevals += len(points)
            ind.set_error_vec(target_values)
            ind.fitness.values = toolbox.evaluate(ind.error_vec)
        # select new individuals according to rules from 'the paper'
        # phenotypic distance
        dist = lambda ind1, ind2: np.linalg.norm(family[family_idx[ind1]].error_vec \
                                               - family[family_idx[ind2]].error_vec)
        tournament = []
        if dist('p1', 'c1') + dist('p2', 'c2') <= dist('p1', 'c2') + dist('p2', 'c1'):
            tournament.append((c1, p1))
            tournament.append((c2, p2))
        else:
            tournament.append((c2, p1))
            tournament.append((c1, p2))
        for ind1, ind2 in tournament:
            if ind1.fitness > ind2.fitness:
                offspring.append(ind1)
            else:
                offspring.append(ind2)

    assert(len(offspring) == len(population))
    # new generation
    return offspring, nevals

def var_and_double_tournament(population, points, toolbox, cxpb, mutpb, fitness_size, parsimony_size):
    offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
    # evaluate
    target_values = np.array([toolbox.target_func(x) for x in points])
    ind_values = np.zeros(len(points))
    invalid_ind = (ind for ind in offspring if not ind.fitness.valid)
    nevals = 0
    for ind in invalid_ind:
        nevals += len(points)
        ind.eval_at_points(points, toolbox.compile)
        ind.set_error_vec(target_values)
        ind.fitness.values = toolbox.evaluate(ind.error_vec)
    return tools.selDoubleTournament(offspring, len(offspring), fitness_size, parsimony_size, False), nevals

def target_func(x):
    return math.exp(abs(x))*math.sin(2 * math.pi * x)

def symbreg_fitness(errors):
    return math.fsum(map(abs, errors)) / len(errors),

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

    last_predictor = None

    # Begin the generational process
    while not _terminate():
        gen += 1
        # get points to use
        predictor = fp.get_best_predictor()
        # if we use new predictor, we have to re-evaluate the population
        if last_predictor is not None and sorted(predictor) != sorted(last_predictor):
            for ind in pop:
                del ind.fitness.values
        last_predictor = predictor
        # crossover, mutation and selection
        offspring, nevals = toolbox.new_gen(population=population, points=points[predictor])
        evals += nevals

        population[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=nevals, **record)

        if verbose:
            print(logbook.stream)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

    return population, logbook

CXPB = 0.5
MUTPB = 0.1
POP_SIZE = 128
POINTS = np.linspace(-3, 3, 200)

def symb_reg_initialize():
    # initialization

    creator.create("Individual", SymbRegTree, fitness=creator.FitnessMin)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", symbreg_fitness)
    toolbox.register("target_func", target_func)
    toolbox.register("new_gen", deterministic_crowding, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB)
    #toolbox.register("new_gen", var_and_double_tournament, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB, fitness_size=3, parsimony_size=1.4)


    # stats we want to keep track of
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats_height = tools.Statistics(lambda ind: ind.height)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size, height=stats_height)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)


    return mstats

if __name__ == '__main__':
    random.seed(69)
    stats = symb_reg_initialize()
    pop = toolbox.population(POP_SIZE)
    hof = tools.HallOfFame(1)
    ngens = 20
    pop, log = symb_reg_with_fp(pop, toolbox, CXPB, MUTPB, 'evals', lambda x: x >= 1e6,
                                fitness_pred.ExactFitness(len(POINTS)), POINTS, stats=stats,
                                halloffame=hof, verbose=False)
    print(hof[0].fitness.values)
