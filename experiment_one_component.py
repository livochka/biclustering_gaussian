from utils.group_estimators import GroupEstimatorNumerical, GroupEstimatorHierarchical, GroupEstimatorGreedy
import numpy as np
import timeit
import pandas as pd


def generate_negative(n_rows):
    mu = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    cov = np.array([[4.5, -2, 1, 0, 0, 0, 0, 0],
                    [-2, 4.5, 2, 0, 0, 0, 0, 0],
                    [1, 2, 4.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4.5, -2, 2, 0, 0],
                    [0, 0, 0, -2, 4.5, 2, 0, 0],
                    [0, 0, 0, 2, 2, 4.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 3, 2],
                    [0, 0, 0, 0, 0, 0, 2, 4.5]])
    X = np.random.multivariate_normal(mean=mu, cov=cov, size=n_rows)
    return X, cov


def generate_positive(n_rows):
    mu = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    cov = np.array([[4.5, 2, 2, 0, 0, 0, 0, 0],
                    [2, 4.5, 2, 0, 0, 0, 0, 0],
                    [2, 2, 4.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4.5, 2, 2, 0, 0],
                    [0, 0, 0, 2, 4.5, 2, 0, 0],
                    [0, 0, 0, 2, 2, 4.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4.5, 2],
                    [0, 0, 0, 0, 0, 0, 2, 4.5]])
    X = np.random.multivariate_normal(mean=mu, cov=cov, size=n_rows)
    return X, cov


def get_true_groups():
    groups = []

    for i in range(3):
        groups.append(0)

    for i in range(3, 6):
        groups.append(1)

    for i in range(6, 8):
        groups.append(2)

    return np.array(groups)


def estimate_parameters(estimator, data, true_groups):
    start = timeit.default_timer()
    estimator.fit(data)
    stop = timeit.default_timer()
    return stop - start, int(estimator.is_correct(true_groups))


def compare_estimators(data_gen_func, n_rows: int, estimators: dict, N=100):
    exec_time = {"hierarchical": [], "numerical": [], "greedy": []}
    is_correct = {"hierarchical": [], "numerical": [], "greedy": []}
    true_groups = get_true_groups()

    for exp in range(N):
        data, _ = data_gen_func(n_rows)
        for est_name in estimators:
            exec_sec, correct = estimate_parameters(estimators[est_name], data, true_groups)
            exec_time[est_name].append(exec_sec)
            is_correct[est_name].append(correct)

    return exec_time, is_correct


if __name__ == "__main__":

    N = 200
    data_func = generate_negative
    n_rows = 80
    folder = "experiment_one_component_results/"

    exp_name = f"one_comp_{data_func.__name__}_rows_{n_rows}_N_{N}"
    estimators = {"numerical": GroupEstimatorNumerical(n_clusters=3, gamma=0.001, lambd=0.0001, conv_epsilon=0.1, step=0.1),
                  "greedy": GroupEstimatorGreedy(n_clusters=3),
                  "hierarchical": GroupEstimatorHierarchical(n_clusters=3)}

    time_result, accuracy_result = compare_estimators(data_func, n_rows, estimators, N)

    time_result = pd.DataFrame(time_result)
    time_result.to_csv(folder + exp_name + "_time.csv")

    accuracy_result = pd.DataFrame(accuracy_result)
    accuracy_result.to_csv(folder + exp_name + "_accuracy.csv")

