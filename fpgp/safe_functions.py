import math

def safe_div(a, b):
    if b == 0: return 1
    return a / b

def safe_exp(x):
    if x >= 20: return math.exp(30)
    if x <= -20: return math.exp(-30)
    return math.exp(x)

def safe_log(x):
    return math.log(max(1e-20, abs(x)))
