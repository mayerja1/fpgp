import random
import math
import numpy as np
import copy
from collections import deque


class FitnessPredictor():

    def __init__(self, training_set_size, size, test_cases=None):
        self._training_set_size = training_set_size
        self._test_cases = test_cases
        self._size = size
        if test_cases is None:
            self.random_predictor()

    @property
    def training_set_size(self):
        return self._training_set_size

    @property
    def test_cases(self):
        return self._test_cases

    @property
    def size(self):
        return self._size

    def random_predictor(self):
        self._test_cases = np.random.randint(self.training_set_size, size=self.size)

    def __str__(self):
        return str(self.test_cases)


class EvolvingFitnessPredictor(FitnessPredictor):

    def __init__(self, training_set_size, size, mutpb, cxpb, test_cases=None):
        super().__init__(training_set_size, size, test_cases=test_cases)
        self.cxpb = cxpb
        self.mutpb = mutpb

    def mutate(self):
        for i in range(self.size):
            if random.random() < self.mutpb:
                self.test_cases[i] = random.randint(0, self.training_set_size - 1)

    def crossover(self, other):
        if self.size != other.size:
            raise ValueError('predictors must have same size')
        if random.random() < self.cxpb:
            xo_point = random.randint(0, self.size - 1)
            self.test_cases[:xo_point] = other.test_cases[:xo_point]


class AdaptiveSizeFitnessPredictor(EvolvingFitnessPredictor):
    pass


class FitnessPredictorManager():

    def __init__(self, training_set_size):
        self.training_set_size = training_set_size

    def get_best_predictor(self):
        raise NotImplementedError()

    def next_generation(self, **kwargs):
        raise NotImplementedError()


class ExactFitness(FitnessPredictorManager):

    def __init__(self, training_set_size):
        super().__init__(training_set_size)
        self.points = np.arange(self.training_set_size, dtype=np.int32)

    def get_best_predictor(self):
        return self.points

    def next_generation(self, **kwargs):
        return 0


class SchmidtLipsonFPManager(FitnessPredictorManager):

    def __init__(self, training_set_size, predictors_size=8, num_predictors=8,
                 mutpb=0.1, cxpb=0.5, num_trainers=10, effort_tresh=0.05, new_trainer_period=100, **kwargs):
        self.predictor_pop = [EvolvingFitnessPredictor(training_set_size, predictors_size, mutpb, cxpb)
                              for _ in range(num_predictors)]

        self.trainers_pop = [None] * num_trainers
        self.trainers_fitness = deque([0] * num_trainers)
        self.pred_evolution_gen = 0
        self.best_pred = None
        self.best_pred_f = -np.inf
        self.effort_tresh = effort_tresh
        self.new_trainer_period = int(new_trainer_period)

    def get_best_predictor(self):
        return self.best_pred.test_cases

    def next_generation(self, **kwargs):
        nevals = 0
        # first call of the function
        if self.trainers_pop[0] is None:
            # random trainers
            self.trainers_pop = deque([copy.deepcopy(random.choice(kwargs['pop'])) for _ in range(len(self.trainers_pop))])
            # get exact fitness
            for i, t in enumerate(self.trainers_pop):
                self.trainers_fitness[i] = \
                    kwargs['toolbox'].individual_fitness(t, kwargs['training_set'],
                                                         kwargs['target_values'],
                                                         kwargs['toolbox'])[0]
                nevals += len(kwargs['training_set'])
        if kwargs['effort'] <= self.effort_tresh:
            #print(kwargs['gen'], self.pred_evolution_gen)
            self.pred_evolution_gen += 1
            nevals += self.deterministic_crowding(kwargs['training_set'], kwargs['target_values'], kwargs['toolbox'])
            # add  new trainer every 100 predictor generations
            if self.pred_evolution_gen % self.new_trainer_period == 0:
                #print('new_trainer')
                nevals += self.add_fitness_trainer(kwargs['pop'], kwargs['training_set'], kwargs['target_values'], kwargs['toolbox'])

        return nevals

    def deterministic_crowding(self, training_set, target_values, toolbox):
        assert(len(self.predictor_pop) % 2 == 0)
        parents = [copy.deepcopy(p) for p in self.predictor_pop]
        offspring = []
        random.shuffle(parents)
        nevals = 0
        for i in range(1, len(parents), 2):
            p1, p2 = parents[i - 1], parents[i]
            c1, c2 = map(copy.deepcopy, (p1, p2))
            c1.crossover(p2)
            c2.crossover(p1)
            c1.mutate()
            c2.mutate()
            # we define distance as number of different test_cases
            dist = lambda x, y: len(set(x.test_cases) ^ set(y.test_cases))
            fitnesses = {p: self.predictor_fitness(p, training_set, target_values, toolbox) for p in (p1, p2, c1, c2)}
            nevals += 4 * len(self.trainers_pop) * len(self.predictor_pop[0].test_cases)
            tournament = []
            if dist(p1, c1) + dist(p2, c2) <= dist(p1, c2) + dist(p2, c1):
                tournament.append((c1, p1))
                tournament.append((c2, p2))
            else:
                tournament.append((c2, p1))
                tournament.append((c1, p2))
            for x, y in tournament:
                if fitnesses[x] > fitnesses[y]:
                    offspring.append(x)
                else:
                    offspring.append(y)
                # if we found new best predictor
                if fitnesses[offspring[-1]] > self.best_pred_f:
                    self.best_pred = copy.deepcopy(offspring[-1])
                    self.best_pred_f = fitnesses[offspring[-1]]

            self.predictor_pop = offspring

        return nevals

    def predictor_fitness(self, p, training_set, target_values, toolbox):
        predicted_fitnesses = [toolbox.individual_fitness(t, training_set[p.test_cases], target_values[p.test_cases], toolbox)[0]
                               for t in self.trainers_pop]
        # negative so that we want to find maximal fitness
        return -math.fsum(map(lambda x: abs(x[0] - x[1]), zip(predicted_fitnesses, self.trainers_fitness))) / len(self.trainers_pop)

    def add_fitness_trainer(self, pop, training_set, target_values, toolbox):
        nevals = 0
        solutions_variances = np.zeros(len(pop))
        for i, s in enumerate(pop):
            predicted_fitnesses = [toolbox.individual_fitness(s, training_set[p.test_cases], target_values[p.test_cases], toolbox)[0]
                                   for p in self.predictor_pop]
            nevals += len(self.predictor_pop) * len(self.predictor_pop[0].test_cases)
            solutions_variances[i] = np.var(predicted_fitnesses)

        # select best one from the population and replace the oldest trainer
        self.trainers_pop.pop()
        self.trainers_fitness.pop()
        new_trainer = copy.deepcopy(pop[np.argmax(solutions_variances)])
        self.trainers_pop.appendleft(new_trainer)
        self.trainers_fitness.appendleft(toolbox.individual_fitness(new_trainer, training_set, target_values, toolbox)[0])
        nevals += len(training_set)
        return nevals


class StaticRandom(FitnessPredictorManager):

    def __init__(self, training_set_size, size=8, **kwargs):
        self.pred = FitnessPredictor(training_set_size, size)

    def get_best_predictor(self):
        return self.pred.test_cases

    def next_generation(self, **kwargs):
        return 0


class DynamicRandom(FitnessPredictorManager):

    def __init__(self, training_set_size, size=8, **kwargs):
        self.pred = FitnessPredictor(training_set_size, size)

    def get_best_predictor(self):
        return self.pred.test_cases

    def next_generation(self, **kwargs):
        self.pred.random_predictor()
        return 0
