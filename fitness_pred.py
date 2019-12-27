import numpy as np
from copy import copy

class FitnessPredictor():

    def __init__(self, number_of_tests, size, test_cases=None):
        self.number_of_tests = number_of_tests
        self.test_cases = test_cases
        self.size = size
        if test_cases is None:
            self.random_predictor()

    def random_predictor(self):
        self.test_cases = np.random.randint(self.number_of_tests, size=self.size)

    def __str__(self):
        return str(self.test_cases)

class EvolvingFitnessPredictor(FitnessPredictor):

    def __init__(self, number_of_tests, size, prob_mutation, prob_xo, test_cases=None):
        super().__init__(number_of_tests, size, test_cases=test_cases)
        self.prob_xo = prob_xo
        self.prob_mutation = prob_mutation

    def mutate(self):
        for i in range(self.size):
            if random() < self.prob_mutation:
                self.test_cases[i] = randint(0, self.number_of_tests - 1)

    def crossover(self, other):
        if self.size != other.size:
            raise ValueError('predictors must have same size')
        if random() < self.prob_xo:
            xo_point = randint(0, self.size - 1)
            self.test_cases[:xo_point] = other.test_cases[:xo_point]

class FitnessPredictorManager():

    def __init__(self, dataset_size):
        self.dataset_size = dataset_size

    def get_best_predictor(self):
        raise NotImplementedError()

    def next_generation(self, **kwargs):
        raise NotImplementedError()

class ExactFitness(FitnessPredictorManager):

    def __init__(self, dataset_size):
        super().__init__(dataset_size)
        self.points = np.arange(self.dataset_size, dtype=np.int32)

    def get_best_predictor(self):
        return self.points, 0

    def next_generation(self, **kwargs):
        pass
