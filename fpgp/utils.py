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
                    log = {'logbook': pickle.load(fp), 'path': os.path.join(self.path, l)}
                yield log
            except IsADirectoryError:
                print('there are more folders in path, are you sure this is the right path?')


def get_results(path):
    logs, names = [], []
    for fname in os.listdir(path):
        if fname.startswith('.'):
            continue
        full_path = os.path.join(path, fname)
        if os.path.isdir(os.path.join(path, fname)):
            logs.append(LoadedLogs(full_path))
            names.append(fname)
    return tuple(logs), tuple(names)


def create_1d_dataset(fname, func, description, x1, x2, trn_points, tst_add_points):
    create_nd_dataset(fname, func, description, [(x1, x2)], trn_points, tst_add_points)


def create_nd_dataset(fname, func, description, intervals, trn_points, tst_add_points):

    def random_points(n):
        points = np.zeros((n, len(intervals)))
        for i, int_ in enumerate(intervals):
            points[:, i] = np.random.rand(n) * (int_[1] - int_[0]) + int_[0]
        return points

    ranges = [np.linspace(*int, trn_points) for int in intervals]
    trn_x = np.array(np.meshgrid(*ranges)).T.reshape(-1, len(ranges))
    trn_y = func(*trn_x.T)

    trn = np.column_stack([trn_x, trn_y])

    tst_x = np.row_stack([trn_x, random_points(tst_add_points)])
    tst_y = func(*tst_x.T)

    tst = np.column_stack([tst_x, tst_y])

    np.savez(fname, trn=trn, tst=tst, description=description)


def get_statistic_from_logs(stat_func, select_func, logs):
    return stat_func([select_func(l) for l in logs])


def save_stats(fname, benchmarks):
    stats = {}
    for exp_dir in benchmarks:
        logs, names = get_results(exp_dir)
        meds = []
        avgs = []
        for l, name in zip(logs, names):
            med = get_statistic_from_logs(np.median, lambda l: l['logbook'].select('test_set_f')[-1], l)
            avg = get_statistic_from_logs(np.mean, lambda l: l['logbook'].select('test_set_f')[-1], l)
            meds.append(med)
            avgs.append(avg)
        stats[exp_dir.split('/')[-1]] = {'meds': meds, 'avgs': avgs}

    with open(fname, 'wb') as f:
        stats = pickle.dump(stats, f)


if __name__ == '__main__':
    #create_dataset('datasets/f1', lambda x: x**2 - x**3, 'f(x) = x^2 - x^3', -5, 5)
    #create_dataset('datasets/f2', lambda x: np.exp(np.abs(x))*np.sin(2*np.pi*x), 'f(x) = e^|x| * sin(2*PI*x)', -3, 3)
    create_1d_dataset('datasets/f6', lambda x: np.exp(np.abs(x))*np.sin(x), 'f(x) = e^|x| * sin(x)', -10, 10, trn_points=200, tst_add_points=200)
    #create_dataset('datasets/f3', lambda x: x**2*np.exp(np.sin(x)) + x + np.sin(np.pi/4 - x**3), 'f(x) = x^2 * e^sin(x) + x + sin(PI/4 - x^3)', -10, 10)
    #create_dataset('datasets/f4', lambda x: np.exp(-x) * x**3 * np.sin(x) * np.cos(x) * (np.sin(x)**2 * np.cos(x) - 1), 'f(x) = e^(-x) * x^3 * sin(x) * cos(x) * (sin(x)^2 * cos(x) - 1)', 0, 10)
    #create_dataset('datasets/f5', lambda x: 10 / ((x - 3)**2 + 5), 'f(x) = 10 / ((x - 3)^2 + 5)', -2, 8)
    #create_dataset('datasets/moje', lambda x: (x - 1000)**2 + 1000, '(x - 1000)^2 + 1000', -10, 10)
    #create_nd_dataset('datasets/f6', lambda x, y: np.sin(x * y) + (x**2 - y**2) * np.exp(x) / (np.abs(np.log(np.abs(y))) + 1), 'f(x, y) = sin(xy) + (x^2 - y^2) * e^x / (|ln(|y|)| + 1)', [(-10, 10), (-10, 10)], 50, 1000)
    #create_nd_dataset('datasets/f6', lambda x, y: np.exp(-0.5*(x**2 + y**2)) / np.sqrt(4*np.pi**2), '2d normal distribution', [(-5, 5), (-5, 5)], 80, 200)
    pass
