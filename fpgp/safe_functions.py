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


if __name__ == '__main__':
    from operator import add, mul
    from math import cos, sin
    f = lambda ARG0, ARG1, ARG2, ARG3, ARG4, ARG5: mul(ARG5, safe_exp(add(add(add(ARG5, ARG5), add(ARG5, cos(sin(safe_log(ARG5))))), add(safe_exp(add(ARG5, safe_div(ARG5, ARG1))), safe_log(ARG5)))))
    print(f(0.0, 0.600, 4.78, 4.24, 3.15, 0.425))
