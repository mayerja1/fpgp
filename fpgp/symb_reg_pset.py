import math
import operator
import safe_functions as sf


from deap import gp


def init_pset(arity):

    pset = gp.PrimitiveSet("MAIN", arity)

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(sf.safe_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(sf.safe_exp, 1)
    pset.addPrimitive(sf.safe_log, 1)

    pset.addTerminal("pi", math.pi)
    pset.addTerminal("0", 0)
    pset.addTerminal("1", 1)
    pset.addTerminal("-1", -1)

    return pset
