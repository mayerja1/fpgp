import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import animation

def visualize_run(training_set, target_func, log, step=1, freq=100):
    fig, ax = plt.subplots()

    ngens = len(log)

    train_vals = list(map(target_func, training_set))

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
        pred_plot.set_data(training_set[predictors[i]], list(map(target_func, training_set[predictors[i]])))
        sol_plot.set_data(training_set, solutions_vals[i])
        ax.legend()
        return pred_plot, target_plot, sol_plot

    ani = FuncAnimation(fig, update, frames=(i for i in range(0, ngens, step)),
                        init_func=init, interval=1/freq * 1000)
    return ani

def predictor_histogram(training_set, target_func, log):
    predictors = log.select('predictor')

    used_tests_idxs = np.concatenate(predictors)
    used_tests = training_set[used_tests_idxs]
    hist, bin_edges = np.histogram(used_tests, bins=len(training_set))

    fig, ax1 = plt.subplots()
    ax1.bar(bin_edges[:-1], hist, width=training_set[1] - training_set[2], alpha=0.5, align='edge')
    ax1.set_ylabel('point usage')
    ax1.legend(['usage'], loc=2)


    ax2 = ax1.twinx()
    ax2.plot(training_set, list(map(target_func, training_set)), linestyle=' ', marker='o', markersize=3,
             color='black', markeredgecolor='white', markeredgewidth=0.1)

    ax2.legend(['f(x)'], loc=1)


if __name__ == '__main__':
    import pickle
    with open('results10.p', 'rb') as f:
        log = pickle.load(f)
        visualize_run(np.linspace(-3, 3, 200), lambda x: np.exp(np.abs(x))*np.sin(2*np.pi*x), log)
