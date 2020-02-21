import operator

from symb_reg_pset import pset

from deap import creator
from deap import base
from deap import gp

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)

toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

_MAX_HEIGHT = 8

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=_MAX_HEIGHT))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=_MAX_HEIGHT))

if __name__ == '__main__':
    c = creator.FitnessMin((2,))
    print(c.values)
