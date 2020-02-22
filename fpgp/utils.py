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


class LoadedLogs:

    def __init__(self, path):
        self._path = path

    def __iter__(self):
        yield from self.get_logs_from_folder()

    def __len__(self):
        len_ = 0
        for name in os.listdir(self.path):
            if name.split('.')[-1] != 'p':
                continue
            len_ += 1
        return len_

    @property
    def path(self):
        return self._path

    def get_logs_from_folder(self):
        for l in os.listdir(self.path):
            if l.split('.')[-1] != 'p':
                continue
            try:
                with open(os.path.join(self.path, l), 'rb') as fp:
                    log = pickle.load(fp)
                yield log
            except IsADirectoryError:
                print('there are more folders in path, are you sure this is the right path?')


def create_dataset(fname, func):
    x1 = -3
    x2 = 3
    trn_x = np.linspace(x1, x2, 200)
    trn_y = func(trn_x)
    tst_x = np.concatenate([trn_x, np.random.random(200) * (x2 - x1) + x1])
    tst_y = func(tst_x)
    trn = np.column_stack([trn_x, trn_y])
    tst = np.column_stack([tst_x, tst_y])
    np.savez(fname, trn=trn, tst=tst)


if __name__ == '__main__':
    create_dataset('datasets/f2', lambda x: np.exp(np.abs(x)) * np.sin(2*np.pi*x))
