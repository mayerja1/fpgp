#!/usr/bin/env python3
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import random
import operator
import traceback
import copy
import json
import os
import pickle
import functools
import uuid

from symb_reg_pset import pset
from symb_reg_toolbox import toolbox
import fitness_pred

from deap import gp
from deap import algorithms
from deap import tools
from deap import algorithms
from deap import creator

# constants used by gp
CXPB = 0.5
MUTPB = 0.1
POP_SIZE = 128

# used dataset, trn = train, tst = test, x = input, y = output
trn_x = trn_y = tst_x = tst_y = None


class SymbRegTree(gp.PrimitiveTree):

    '''
    list of atributes that aren't copied during deepcopy
    it is a significant speedup of the program
    there is no reason to copy these attributes because
    for individuals with invalid fitness they are computed again
    '''
    deepcopy_ignorelist = ('values', 'error_vec')

    def __init__(self, content):
        super().__init__(content)
        self.values = []
        self.error_vec = None
        self.func = None

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        for k, v in self.__dict__.items():
            if k in SymbRegTree.deepcopy_ignorelist:
                setattr(new, k, v)
            else:
                setattr(new, k, copy.deepcopy(v))
        return new

    def make_invalid(self):
        del self.fitness.values
        self.func = None

    def set_func(self, compile_func):
        self.func = compile_func(self)

    def set_values(self, points, compile_func):
        self.set_func(compile_func)
        self.values = [self.func(p) for p in points]

    def set_error_vec(self, target_values):
        assert(len(self.values) == len(target_values))
        self.error_vec = target_values - self.values

    def eval_at_points(self, points):
        assert(self.func is not None)
        return [self.func(p) for p in points]


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
        c1.make_invalid()
        c2.make_invalid()
        # p1 and p2 are crossed over, we need to get back to their originals
        p1, p2 = parents[i - 1], parents[i]

        # if we weren't supposed to crossover, use the original parent
        if random.random() > cxpb:
            c1 = toolbox.clone(p1)
        if random.random() > cxpb:
            c2 = toolbox.clone(p2)

        # mutate
        if random.random() < mutpb:
            c1, = toolbox.mutate(c1)
            c1.make_invalid()
        if random.random() < mutpb:
            c2, = toolbox.mutate(c2)
            c2.make_invalid()
        family = (p1, p2, c1, c2)
        invalid_ind = (ind for ind in family if not ind.fitness.valid)
        family_idx = {'p1': 0, 'p2': 1, 'c1': 2, 'c2': 3}
        # selection
        # evaluate, get errors and fitnesses
        target_values = np.array([toolbox.target_func(x) for x in points])
        for ind in invalid_ind:
            ind.set_values(points, toolbox.compile)
            nevals += len(points)
            ind.set_error_vec(target_values)
            ind.fitness.values = toolbox.evaluate(ind.error_vec)
        # select new individuals according to rules from 'the paper'
        # phenotypic distance
        dist = lambda ind1, ind2: np.linalg.norm(family[family_idx[ind1]].error_vec
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
        ind.set_values(points, toolbox.compile)
        ind.set_error_vec(target_values)
        ind.fitness.values = toolbox.evaluate(ind.error_vec)
    return tools.selDoubleTournament(offspring, len(offspring), fitness_size, parsimony_size, False), nevals


@functools.lru_cache(maxsize=400)
def target_func(x):
    # simpy return y-values from training/testing data
    output_vals = np.concatenate([trn_y[trn_x == x], tst_y[tst_x == x]])
    assert(len(output_vals) > 0)
    return output_vals[0]


def symbreg_fitness(errors):
    return math.fsum(map(abs, errors)) / len(errors),


def individual_fitness(ind, points, target, toolbox):
    '''
    get fitness of an individual without changing its inner state
    individuals set_func() has to have been called before
    '''
    vals = ind.eval_at_points(points)
    return toolbox.evaluate(target - vals)


def symb_reg_with_fp(population, toolbox, cxpb, mutpb, end_cond, end_func, fp, training_set, test_set, halloffame,
                     stats=None, verbose=__debug__, epsilon=1e-3):

    def _terminate():
        # check convergence
        if len(halloffame) > 0:
            nonlocal evals, best_sol_vals
            errors = map(abs, train_set_target - best_sol_vals)
            max_error = max(errors)

            if max_error < epsilon:
                return True
        # check given condition
        vars_name_mapping = {'gen': gen, 'evals': evals, 'time': time.time() - start_time,
                             'best_fitness': halloffame[0].fitness.values if len(halloffame) > 0 else np.inf}
        return end_func(vars_name_mapping[end_cond])

    logbook = tools.Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])

    gen = 0
    evals = 0
    pred_evals = 0
    start_time = time.time()

    last_predictor = None

    test_set_target = np.array([toolbox.target_func(p) for p in test_set])
    train_set_target = np.array([toolbox.target_func(p) for p in training_set])

    # set functions of first generation, needed for predictor initialization
    for ind in population:
        ind.set_func(toolbox.compile)

    # Begin the generational process
    while not _terminate():
        print('{:.2e}'.format(evals), end='\r')
        gen += 1
        # get points to use
        nevals = fp.next_generation(gen=gen, pop=population, training_set=training_set,
                                    target_values=train_set_target, toolbox=toolbox,
                                    effort=pred_evals / evals if evals != 0 else 0)

        predictor = fp.get_best_predictor()
        pred_evals += nevals
        evals += nevals
        # if we use new predictor, we have to re-evaluate the population
        if last_predictor is not None and sorted(predictor) != sorted(last_predictor):
            for ind in population:
                ind.make_invalid()
        last_predictor = predictor
        # crossover, mutation and selection
        offspring, nevals = toolbox.new_gen(population=population, points=training_set[predictor])
        evals += nevals

        population[:] = offspring

        # Update the hall of fame with the generated individuals and gen values of the best individual
        halloffame.update(population)
        best_sol_vals = halloffame[0].eval_at_points(training_set)
        evals += len(training_set)

        # get test_set fitness
        best_ind = halloffame[0]
        vals = best_ind.eval_at_points(test_set)
        test_set_f = toolbox.evaluate(test_set_target - vals)[0]

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, evals=evals, test_set_f=test_set_f, predictor=predictor, best_sol_vals=best_sol_vals, **record)

        if verbose:
            print(logbook.stream)

    return population, logbook


_toolbox_registered = False


def symb_reg_initialize():
    # initialization
    global _toolbox_registered
    if not _toolbox_registered:
        creator.create("Individual", SymbRegTree, fitness=creator.FitnessMin)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)
        toolbox.register("evaluate", symbreg_fitness)
        toolbox.register("individual_fitness", individual_fitness)
        toolbox.register("target_func", target_func)
        toolbox.register("new_gen", deterministic_crowding, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB)
        #toolbox.register("new_gen", var_and_double_tournament, toolbox=toolbox, cxpb=CXPB, mutpb=MUTPB, fitness_size=3, parsimony_size=1.4)
        _toolbox_registered = True

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


def run(end_cond='gen', end_func=lambda x: x >= 1000, fitness_predictor='exact', epsilon=1e-3):
    stats = symb_reg_initialize()
    pop = toolbox.population(POP_SIZE)
    hof = tools.HallOfFame(1)

    predictors = {'exact': fitness_pred.ExactFitness(len(trn_x)),
                  'SchmidtLipson': fitness_pred.SchmidtLipsonFPManager(len(trn_x), predictors_size=8)}

    pop, log = symb_reg_with_fp(pop, toolbox, CXPB, MUTPB, end_cond, end_func,
                                predictors[fitness_predictor], trn_x, tst_x, halloffame=hof,
                                stats=stats, verbose=False, epsilon=epsilon)

    return pop, log, hof


def load_dataset(fname):
    x = np.load(fname)
    global trn_x, trn_y, tst_x, tst_y
    trn_x, trn_y = x['trn_x'], x['trn_y']
    tst_x, tst_y = x['tst_x'], x['tst_y']


def run_config(fname):
    with open(fname, 'r') as fp:
        cfg = json.load(fp)
    for experiment in cfg:
        print(f'starting experiment: {experiment["name"]}')
        load_dataset(experiment['dataset'])
        # make end func callable
        experiment['run_args']['end_func'] = eval(experiment['run_args']['end_func'])
        path = f'data/{experiment["name"]}'
        # copy dataset, so that we know which dataset was used
        np.savez(f'{path}/dataset.npz', trn_x=trn_x, trn_y=trn_y, tst_x=tst_x, tst_y=tst_y)
        try:
            os.makedirs(path)
        except FileExistsError:
            print('the experiment folder already exists...')
        for i in range(experiment['runs']):
            print(f'starting run {i}')
            _, log, _ = run(**experiment['run_args'])
            with open(path + f'{uuid.uuid4()}.p', 'wb') as fp:
                pickle.dump(log, fp)


if __name__ == '__main__':
    run_config('experiments/f2_1e7_evals.json')
    #load_dataset('datasets/f1.npz')
    #run(fitness_predictor='SchmidtLipson')
