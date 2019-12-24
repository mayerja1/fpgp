import math

def safe_div(a, b):
    if b == 0: return 1
    return a / b

def safe_exp(x):
    return math.exp(min(20, x))

def safe_log(x):
    return math.log(max(1e-20, abs(x)))
