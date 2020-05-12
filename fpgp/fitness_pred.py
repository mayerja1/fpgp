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
        self._cxpb = cxpb
        self._mutpb = mutpb

    @property
    def cxpb(self):
        return self._cxpb

    @property
    def mutpb(self):
        return self._mutpb

    def mutate(self):
        for i in range(self.size):
            if random.random() < self.mutpb:
                self._test_cases[i] = random.randint(0, self.training_set_size - 1)

    def crossover(self, other):
        if self.size != other.size:
            raise ValueError('predictors must have same size')
        if random.random() < self.cxpb:
            xo_point = random.randint(0, self.size - 1)
            self._test_cases[:xo_point] = other._test_cases[:xo_point]


class AdaptiveSizeFitnessPredictor(EvolvingFitnessPredictor):

    def __init__(self, training_set_size, size, mutpb, cxpb, init_read_length, test_cases=None):
        self.read_length = init_read_length
        super().__init__(training_set_size, size, mutpb, cxpb, test_cases=test_cases)
        self._cxpb = cxpb
        self._mutpb = mutpb

    @property
    def test_cases(self):
        # use set to remove duplicates
        return np.array(list(set(self._test_cases[:self.read_length])))

    def crossover(self, other):
        if self.size != other.size:
            raise ValueError('predictors must have same size')
        if random.random() < self.cxpb:
            xo_point = random.randint(0, self.read_length - 1)
            self._test_cases[:xo_point] = other._test_cases[:xo_point]


class FitnessPredictorManager():
    '''
    Base class to use fitness predictors during the evolution
    To implement your own method of constructing the predictors,
    derive from this class and implement following methods.
    '''

    def __init__(self, training_set_size):
        self.training_set_size = training_set_size

    def get_best_predictor(self):
        '''
        Return the predictors that will be used by GP in the current generation.
        Return type: numpy array (n,), dtype=int
        '''
        raise NotImplementedError()

    def next_generation(self, **kwargs):
        '''
        This method is run every generation. If inner state of the class needs
        to be updated (i.e. to a find better predictor), this is a way to do it.
        kwargs contain useful info about current state of the evolution, for example
        current generation, current population etc.
        '''
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

    def __init__(self, training_set_size, pred_size=32, num_predictors=8,
                 mutpb=0.1, cxpb=0.5, num_trainers=10, effort_tresh=0.05, new_trainer_period=100, **kwargs):
        self.predictor_pop = [EvolvingFitnessPredictor(training_set_size, pred_size, mutpb, cxpb)
                              for _ in range(num_predictors)]

        self.trainers_pop = [None] * num_trainers
        self.trainers_fitness = deque([0] * num_trainers)
        self.pred_evolution_gen = 0
        self.best_pred = None
        self.best_pred_f = -np.inf
        self.effort_tresh = effort_tresh
        self.new_trainer_period = int(new_trainer_period)

    def get_best_predictor(self):
        #print(self.best_pred_f)
        return self.best_pred.test_cases

    def next_generation(self, **kwargs):
        # first call of the function
        if self.trainers_pop[0] is None:
            # random trainers
            self.trainers_pop = deque([copy.deepcopy(random.choice(kwargs['pop'])) for _ in range(len(self.trainers_pop))])
            # get exact fitness
            for i, t in enumerate(self.trainers_pop):
                self.trainers_fitness[i] = \
                    kwargs['toolbox'].individual_fitness(t, kwargs['training_set'],
                                                         kwargs['target_values'],
                                                         kwargs['toolbox'])
        if kwargs['effort'] <= self.effort_tresh:
            self.pred_evolution_gen += 1
            self.deterministic_crowding(kwargs['training_set'], kwargs['target_values'], kwargs['toolbox'])
            # add  new trainer every 100 predictor generations
            if self.pred_evolution_gen % self.new_trainer_period == 0:
                self.add_fitness_trainer(kwargs['pop'], kwargs['training_set'], kwargs['target_values'], kwargs['toolbox'])

    def deterministic_crowding(self, training_set, target_values, toolbox):
        assert(len(self.predictor_pop) % 2 == 0)
        parents = [copy.deepcopy(p) for p in self.predictor_pop]
        offspring = []
        random.shuffle(parents)
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

    def predictor_fitness(self, p, training_set, target_values, toolbox):
        predicted_fitnesses = [toolbox.individual_fitness(t, training_set[p.test_cases], target_values[p.test_cases], toolbox)
                               for t in self.trainers_pop]
        # negative because we want to find maximal fitness
        return -math.fsum(map(lambda x: abs(toolbox.fitness_diff(x[0], x[1])[0]), zip(predicted_fitnesses, self.trainers_fitness))) / len(self.trainers_pop)

    def add_fitness_trainer(self, pop, training_set, target_values, toolbox):
        solutions_variances = np.zeros(len(pop))
        for i, s in enumerate(pop):
            predicted_fitnesses_vals = [toolbox.individual_fitness(s, training_set[p.test_cases], target_values[p.test_cases], toolbox).values[0]
                                        for p in self.predictor_pop]
            solutions_variances[i] = np.var(predicted_fitnesses_vals)

        # select best one from the population and replace the oldest trainer
        self.trainers_pop.pop()
        self.trainers_fitness.pop()
        new_trainer = copy.deepcopy(pop[np.argmax(solutions_variances)])
        self.trainers_pop.appendleft(new_trainer)
        self.trainers_fitness.appendleft(toolbox.individual_fitness(new_trainer, training_set, target_values, toolbox))


class DrahosovaSekaninaFPManager(FitnessPredictorManager):

    def __init__(self, training_set_size, num_predictors=32, mutpb=0.2, cxpb=0.5,
                 num_trainers=16, generation_period=100, init_read_length=5, **kwargs):
        super().__init__(training_set_size)
        self.predictor_pop = [AdaptiveSizeFitnessPredictor(training_set_size, training_set_size, mutpb, cxpb, init_read_length)
                              for _ in range(num_predictors)]
        self.trainers_pop = deque([None] * num_trainers)
        self.trainers_objective_f = deque([None] * num_trainers)
        self.best_pred = None
        self.best_pred_f = -np.inf
        self._read_length = init_read_length
        self.generation_period = generation_period

        self.last_read_length_update_gen = 0
        self.last_read_length_update_objective_fitness = None

        self.last_gen_subjective_fitness = None

        self.training_set_size = training_set_size

    @property
    def read_length(self):
        return self._read_length

    @read_length.setter
    def read_length(self, value):
        self._read_length = value
        for p in self.predictor_pop:
            p.read_length = self._read_length
        self.best_pred.read_length = self._read_length

    def get_best_predictor(self):
        return self.best_pred.test_cases

    def next_generation(self, **kwargs):
        # not initialized
        if self.trainers_pop[0] is None:
            self.trainers_pop = deque([copy.deepcopy(random.choice(kwargs['pop'])) for _ in range(len(self.trainers_pop))])
            # TODO: this could be a one-liner
            for i, t in enumerate(self.trainers_pop):
                self.trainers_objective_f[i] = \
                    kwargs['toolbox'].individual_fitness(t, kwargs['training_set'],
                                                         kwargs['target_values'],
                                                         kwargs['toolbox'])

        # time to perform next generation (== 1 is because first is generation 1
        # and we need to run this in order to initialize)
        if kwargs['gen'] % self.generation_period == 1:
            selected = self.tournament_selection(len(self.trainers_pop) * 2, kwargs['training_set'],
                                                 kwargs['target_values'], kwargs['toolbox'])
            for i in range(0, len(selected) - 1, 2):
                p1, p2 = selected[i], selected[i + 1]
                p1.crossover(p2)
                p1.mutate()
                self.predictor_pop[i // 2] = p1

        sub_f = kwargs['toolbox'].individual_fitness(kwargs['best_solution'], kwargs['training_set'][self.get_best_predictor()],
                                                     kwargs['target_values'][self.get_best_predictor()],
                                                     kwargs['toolbox'])
        obj_f = None
        # time to update read_length
        if self.last_gen_subjective_fitness is None \
           or sub_f > self.last_gen_subjective_fitness \
           or kwargs['gen'] - self.last_read_length_update_gen >= 1000:
            obj_f = kwargs['toolbox'].individual_fitness(kwargs['best_solution'], kwargs['training_set'],
                                                         kwargs['target_values'],
                                                         kwargs['toolbox'])
            if kwargs['gen'] > 1:
                velocity = kwargs['toolbox'].fitness_diff(obj_f, self.last_read_length_update_objective_fitness)[0] / (kwargs['gen'] - self.last_read_length_update_gen)
            else:
                velocity = 1

            inaccuracy = sub_f.values[0] / obj_f.values[0]
            self.update_read_length(velocity, inaccuracy)

            self.last_read_length_update_gen = kwargs['gen']
            self.last_read_length_update_objective_fitness = copy.deepcopy(obj_f)
        self.last_gen_subjective_fitness = copy.deepcopy(sub_f)

        # check if we want to add a new trainer
        # get subjective fitness of the whole trainers population
        trainers_sub_f = [kwargs['toolbox'].individual_fitness(t, kwargs['training_set'][self.get_best_predictor()],
                                                               kwargs['target_values'][self.get_best_predictor()],
                                                               kwargs['toolbox'])
                          for t in self.trainers_pop]
        # if the current best solution is better than best trainer
        if all(sub_f > f for f in trainers_sub_f):
            # compute objective fitness if necessary
            if obj_f is None:
                obj_f = kwargs['toolbox'].individual_fitness(kwargs['best_solution'], kwargs['training_set'],
                                                             kwargs['target_values'],
                                                             kwargs['toolbox'])
            # replace oldest trainer
            self.add_fitness_trainer(kwargs['best_solution'], obj_f)

    def update_read_length(self, velocity, inaccuracy):
        '''if inaccuracy > 2.7:
            c = 1.2
        elif abs(velocity) <= 0.001:
            c = 0.9
        elif velocity < 0:
            c = 0.96
        elif 0 < velocity <= 0.1:
            c = 1.07
        else:
            c = 1.0'''
        # over-fitting
        if inaccuracy < 1 / 1.75:
            c = 1.2
        # stagnation
        elif abs(velocity) <= 0.001:
            c = 0.9
        # detoriation
        elif velocity > 0:
            c = 0.96
        # improvement
        elif -0.1 <= velocity < 0:
            c = 1.07
        else:
            c = 1.0
        next_read_length = int(round(self.read_length * c))
        next_read_length = max(5, next_read_length)
        next_read_length = min(next_read_length, self.training_set_size)

        self.read_length = next_read_length

    def add_fitness_trainer(self, t, obj_f):
        self.trainers_pop.pop()
        self.trainers_objective_f.pop()
        self.trainers_pop.appendleft(t)
        self.trainers_objective_f.appendleft(obj_f)

    def tournament_selection(self, n, training_set, target_values, toolbox, tournament_size=2):
        fitnesses = {p: self.predictor_fitness(p, training_set, target_values, toolbox) for p in self.predictor_pop}
        # update best predictor
        for p, f in fitnesses.items():
            if f > self.best_pred_f:
                self.best_pred_f = copy.deepcopy(f)
                self.best_pred = copy.deepcopy(p)
        ret = []
        for _ in range(n):
            contenders = [random.choice(self.predictor_pop) for _ in range(tournament_size)]
            contenders_f = [fitnesses[p] for p in contenders]
            winner = copy.deepcopy(contenders[np.argmax(contenders_f)])
            ret.append(winner)
        return ret

    def predictor_fitness(self, p, training_set, target_values, toolbox):
        predicted_fitnesses = [toolbox.individual_fitness(t, training_set[p.test_cases], target_values[p.test_cases], toolbox)
                               for t in self.trainers_pop]
        # negative because we want to find maximal fitness
        return -math.fsum(map(lambda x: abs(toolbox.fitness_diff(x[0], x[1])[0]), zip(predicted_fitnesses, self.trainers_objective_f))) / len(self.trainers_pop)


class StaticRandom(FitnessPredictorManager):

    def __init__(self, training_set_size, pred_size=32, **kwargs):
        self.pred = FitnessPredictor(training_set_size, pred_size)

    def get_best_predictor(self):
        return self.pred.test_cases

    def next_generation(self, **kwargs):
        pass


class DynamicRandom(FitnessPredictorManager):

    def __init__(self, training_set_size, pred_size=32, **kwargs):
        self.pred = FitnessPredictor(training_set_size, pred_size)

    def get_best_predictor(self):
        return self.pred.test_cases

    def next_generation(self, **kwargs):
        self.pred.random_predictor()


class MyPred2(FitnessPredictorManager):

    def __init__(self, training_set_size, pred_size=5, period=100, **kwargs):
        self.pred = FitnessPredictor(training_set_size, pred_size)
        self.period = period

        self.last_read_length_update_gen = 0
        self.last_read_length_update_objective_fitness = None

        self.last_gen_subjective_fitness = None

    def get_best_predictor(self):
        #print(len(self.pred.test_cases))
        return self.pred.test_cases

    def next_generation(self, **kwargs):

        sub_f = kwargs['toolbox'].individual_fitness(kwargs['best_solution'], kwargs['training_set'][self.get_best_predictor()],
                                                     kwargs['target_values'][self.get_best_predictor()],
                                                     kwargs['toolbox'])
        new_size = self.pred.size
        sorted_test_cases = None
        # update the size
        if self.last_gen_subjective_fitness is None \
           or sub_f > self.last_gen_subjective_fitness \
           or kwargs['gen'] - self.last_read_length_update_gen >= 1000:

            obj_f = kwargs['toolbox'].individual_fitness(kwargs['best_solution'], kwargs['training_set'],
                                                         kwargs['target_values'],
                                                         kwargs['toolbox'])
            if kwargs['gen'] > 1:
                velocity = kwargs['toolbox'].fitness_diff(obj_f, self.last_read_length_update_objective_fitness)[0] / (kwargs['gen'] - self.last_read_length_update_gen)
            else:
                velocity = 1

            inaccuracy = sub_f.values[0] / obj_f.values[0]
            new_size = self.new_size(velocity, inaccuracy)

            self.last_read_length_update_gen = kwargs['gen']
            self.last_read_length_update_objective_fitness = copy.deepcopy(obj_f)
        self.last_gen_subjective_fitness = copy.deepcopy(sub_f)

        size_diff = new_size - self.pred.size
        if size_diff < 0:
            self.pred = FitnessPredictor(self.pred.training_set_size, new_size, test_cases=self.pred.test_cases[:new_size])
        elif size_diff > 0:
            sorted_test_cases = self.sorted_test_cases_idx_wrt_WMAE(kwargs)
            new_test_cases = np.append(self.pred.test_cases, -np.ones(size_diff, dtype=int))
            test_case_idx = 0
            for i in range(new_size - size_diff, new_size):
                while sorted_test_cases[test_case_idx] in new_test_cases:
                    test_case_idx += 1
                new_test_cases[i] = sorted_test_cases[test_case_idx]
            assert(np.all(new_test_cases >= 0))
            self.pred = FitnessPredictor(self.pred.training_set_size, new_size, new_test_cases)

        # periodically replace 'useless' test cases
        if kwargs['gen'] % self.period == 1:
            if sorted_test_cases is None:
                sorted_test_cases = self.sorted_test_cases_idx_wrt_WMAE(kwargs)
            test_cases_set = set(self.pred.test_cases)
            to_be_replaced = -1
            for x in reversed(sorted_test_cases):
                if x in test_cases_set:
                    tmp, = np.where(self.pred.test_cases == x)
                    assert(len(tmp == 1))
                    to_be_replaced = tmp[0]
            assert(to_be_replaced != -1)
            for i in sorted_test_cases:
                if i not in self.pred.test_cases:
                    self.pred.test_cases[to_be_replaced] = i

    def new_size(self, velocity, inaccuracy):
        # over-fitting
        if inaccuracy < 1 / 1.75:
            c = 1.2
        # stagnation
        elif abs(velocity) <= 0.001:
            c = 0.9
        # detoriation
        elif velocity > 0:
            c = 0.96
        # improvement
        elif -0.1 <= velocity < 0:
            c = 1.07
        else:
            c = 1.0
        #print(c)
        cur_size = self.pred.size
        next_size = int(round(cur_size * c))
        next_size = max(5, next_size)
        next_size = min(next_size, self.pred.training_set_size)
        return next_size

    def sorted_test_cases_idx_wrt_WMAE(self, kwargs):
        population_values = np.array([ind.eval_at_points(kwargs['training_set']) for ind in kwargs['pop']])
        assert(population_values.shape == (len(kwargs['pop']), len(kwargs['training_set'])))
        obj_fitnesses = np.array([kwargs['toolbox'].evaluate(e)[0] for e in population_values - kwargs['target_values']])
        mean_abs_errors_weightened = np.average(np.abs(population_values - kwargs['target_values']), weights=1/obj_fitnesses, axis=0)
        sorted_ = sorted(zip(mean_abs_errors_weightened, range(len(mean_abs_errors_weightened))), reverse=True)
        assert(len(mean_abs_errors_weightened) == len(kwargs['training_set']))
        return np.array([i for _, i in sorted_])
