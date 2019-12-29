import math
import operator
import safe_functions as sf
import random

from deap import gp

pset = gp.PrimitiveSet("MAIN", 1)

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(sf.safe_div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(sf.safe_exp, 1)
pset.addPrimitive(sf.safe_log, 1)

pset.addEphemeralConstant("rand101", lambda: random.randint(-1, 1))
pset.addTerminal("pi", math.pi)
pset.renameArguments(ARG0='x')
