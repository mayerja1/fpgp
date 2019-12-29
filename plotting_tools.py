import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib import animation

def visualize_run(training_set, target_func, log, freq=100):
    fig, ax = plt.subplots()

    ngens = len(log)

    train_vals = list(map(target_func, training_set))

    solutions_vals = log.select('best_sol_vals')
    predictors = log.select('predictor')
    test_set_f = log.select('test_set_f')

    pred_plot, = plt.plot([], [], ls=' ', marker='o')
    target_plot, = plt.plot([], [], color='blue', alpha=0.5)
    sol_plot, = plt.plot([], [], color='red')


    def init():
        ax.set_xlim(min(training_set), max(training_set))
        ax.set_ylim(min(train_vals), max(train_vals))
        target_plot.set_data(training_set, train_vals)
        return pred_plot, target_plot, sol_plot

    def update(i):
        ax.set_title(f'generation: {i}\ntest set fitness: {test_set_f[i]}')
        pred_plot.set_data(training_set[predictors[i]], target_func(training_set[predictors[i]]))
        sol_plot.set_data(training_set, solutions_vals[i])
        return pred_plot, target_plot, sol_plot

    ani = FuncAnimation(fig, update, frames=range(1, ngens),
                        init_func=init, interval=1/freq * 1000)
    plt.show()


if __name__ == '__main__':
    import pickle
    with open('results3.p', 'rb') as f:
        log = pickle.load(f)
        visualize_run(np.linspace(-3, 3, 200), lambda x: np.exp(np.abs(x))*np.sin(2*np.pi*x), log)
