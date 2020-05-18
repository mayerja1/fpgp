import utils
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import animation
import warnings
from functools import partial


_latex_names = {
    'SLcoev': '$CSP$',
    'DScoev': '$ASP$',
    'exact': '$GP_{std}$',
    'dynamic': '$RP_{dynamic}$',
    'static': '$RP_{static}$',
    'my2': '$DP$',
    'MyPred2': '$DP$'
}


def names_to_latex(names):
    return tuple([_latex_names[n] if n in _latex_names else n for n in names])


def visualize_run(training_set, train_vals, log, step=1, freq=100):
    fig, ax = plt.subplots()

    ngens = len(log)

    solutions_vals = log.select('best_sol_vals')
    predictors = log.select('predictor')
    test_set_f = log.select('test_set_f')
    evals = log.select('evals')

    pred_plot, = plt.plot([], [], ls=' ', marker='o', label='used predictor')
    target_plot, = plt.plot([], [], color='blue', alpha=0.5, label='target function')
    sol_plot, = plt.plot([], [], color='red', label='best solution')

    def init():
        ax.set_xlim(min(training_set), max(training_set))
        ax.set_ylim(min(train_vals), max(train_vals))
        target_plot.set_data(training_set, train_vals)
        return pred_plot, target_plot, sol_plot

    def update(i):
        s = '{:.2e}'.format(evals[i])
        ax.set_title(f'generation: {i + 1}, evals: {s}\ntest set fitness: {test_set_f[i]}')
        pred_plot.set_data(training_set[predictors[i]], train_vals[predictors[i]])
        sol_plot.set_data(training_set, solutions_vals[i])
        ax.legend()
        return pred_plot, target_plot, sol_plot

    ani = FuncAnimation(fig, update, frames=(i for i in range(0, ngens, step)),
                        init_func=init, interval=1/freq * 1000)
    return ani


def predictor_histogram(training_set, training_vals, logs, ax1=None):
    predictors = [np.concatenate(log['logbook'].select('predictor')) for log in logs]

    used_tests_idxs = np.concatenate(predictors)
    used_tests = training_set[used_tests_idxs]
    hist, bin_edges = np.histogram(used_tests, bins=len(training_set))

    if ax1 is None:
        fig, ax1 = plt.subplots()
    ax1.bar(bin_edges[:-1], hist, width=training_set[1] - training_set[2], alpha=0.5, align='edge')
    ax1.set_ylabel('point usage')
    ax1.legend(['usage'], loc=2)

    ax2 = ax1.twinx()
    ax2.plot(training_set, training_vals, linestyle=' ', marker='o', markersize=3,
             color='black', markeredgecolor='white', markeredgewidth=0.1)

    ax2.legend(['f(x)'], loc=1)


# methods is a list of lists of dicts
def compare_performance(methods, x, y, min_x=None, max_x=None, num_points=10,
                        method_names=[], xlabel=None, ylabel=None, ignore_tresh=1e6,
                        fig=None, ax=None, title=None, xscale='linear', stat_type='mean',
                        vals_errors_func=lambda x: (np.nanmean(x, axis=0), np.nanstd(x, axis=0)),
                        plot_kwargs=None, legend=True):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    # get range of values
    if min_x is None:
        min_x = np.inf
        for method in methods:
            min_x = min(min_x, np.min([l['logbook'].select(x)[0] for l in method]))
    if max_x is None:
        max_x = -np.inf
        for method in methods:
            max_x = max(max_x, np.max([l['logbook'].select(x)[-1] for l in method]))
    if xscale == 'linear':
        points = np.linspace(min_x, max_x, num=num_points)
    elif xscale == 'log':
        points = np.geomspace(min_x, max_x, num=num_points)
    if plot_kwargs is None:
        plot_kwargs = len(methods) * [{}]
    xlabel = x if xlabel is None else xlabel
    ylabel = y if ylabel is None else ylabel
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale(xscale)
    lines = []
    for method, plot_kw in zip(methods, plot_kwargs):
        xss, yss = utils.get_xss_yss_from_logbooks([l['logbook'] for l in method], x, y)
        vals = utils.vals_at_points(xss, yss, points)
        if vals[vals > ignore_tresh].size > 0:
            warnings.warn(f'{vals[vals > ignore_tresh].size} values were ignored for being too high')
        vals[vals > ignore_tresh] = np.nan
        vals_, errors = vals_errors_func(vals)
        err_cont = ax.errorbar(points, vals_, yerr=errors, capsize=2, **plot_kw)
        lines.append(err_cont.lines[0])
    if legend:
        ax.legend(method_names)
    if title is not None:
        ax.set_title(title)
    return lines


def show_performance(log, x, y):
    compare_performance([[log]], x, y, min_x=min(log['logbook'].select(x)), max_x=max(log['logbook'].select(x)), ignore_tresh=np.inf)


def two_stats_graph(log, stat1, stat2):
    step = len(log.select('gen')) // 10
    fig, ax1 = plt.subplots()
    ax1.plot(log.select('gen')[::step], log.select(stat1)[::step], color='blue')
    ax1.set_ylabel(stat1)
    ax1.legend([stat1], loc=2)
    ax2 = ax1.twinx()
    ax2.plot(log.select('gen')[::step], log.select(stat2)[::step], color='red')
    ax2.set_ylabel(stat2)
    ax2.legend([stat2], loc=1)


if __name__ == '__main__':
    import pickle
    with open('results10.p', 'rb') as f:
        log = pickle.load(f)
        visualize_run(np.linspace(-3, 3, 200), lambda x: np.exp(np.abs(x))*np.sin(2*np.pi*x), log)
