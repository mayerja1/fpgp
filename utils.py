import numpy as np
import os
import pickle


def binary_search(l, v):

    def _binary_search(l, v, a, b):
        if a > b:
            return b, a
        mid = a + (b - a) // 2
        if l[mid] == v:
            return mid, mid
        elif l[mid] > v:
            return _binary_search(l, v, a, mid - 1)
        return _binary_search(l, v, mid + 1, b)

    return _binary_search(l, v, 0, len(l) - 1)


def linear_interpolation(x1, y1, x2, y2, x):
    return y1 + (x - x1) / (x2 - x1) * (y2 - y1)


def val_at_point(xs, ys, x):
    a, b = binary_search(xs, x)
    # the value is there exactly
    if a == b:
        return ys[a]
    # the value lies within xs
    if a >= 0 and b < len(xs):
        return linear_interpolation(xs[a], ys[a], xs[b], ys[b], x)
    # the value is before the first x
    if a < 0:
        return ys[0]
    # the value is after last x
    return ys[-1]


def vals_at_points(xss, yss, points):
    vals = np.zeros((len(xss), len(points)))
    for i, p in enumerate(points):
        for j, (xs, ys) in enumerate(zip(xss, yss)):
            vals[j, i] = val_at_point(xs, ys, p)
    return vals


def get_xss_yss_from_logbooks(logbooks, x, y):
    xss, yss = [], []
    for log in logbooks:
        xss.append(log.select(x))
        yss.append(log.select(y))
    return xss, yss


def get_logs_from_folder(path):
    logs = []
    for l in os.listdir(path):
        try:
            if l.split('.')[-1] != 'p':
                continue
            with open(os.path.join(path, l), 'rb') as fp:
                logs.append(pickle.load(fp))
        except IsADirectoryError:
            print('there are more folders in path, are you sure this is the right path?')
    return logs
