import numpy as np
import pandas as pd
from utils.clusterizers import HierarcicalBiclustering, FactorAnalyzerBiclustering
from sklearn import metrics
from itertools import permutations

COV_POSITIVE = np.array([[4.5, 2, 2, 0, 0, 0, 0, 0],
                    [2, 4.5, 2, 0, 0, 0, 0, 0],
                    [2, 2, 4.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4.5, 2, 2, 0, 0],
                    [0, 0, 0, 2, 4.5, 2, 0, 0],
                    [0, 0, 0, 2, 2, 4.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4.5, 2],
                    [0, 0, 0, 0, 0, 0, 2, 4.5]])

COV_NEGATIVE = np.array([[4.5, -2, 2, 0, 0, 0, 0, 0],
                    [-2, 4.5, 2, 0, 0, 0, 0, 0],
                    [2, 2, 4.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4.5, -2, 2, 0, 0],
                    [0, 0, 0, -2, 4.5, 2, 0, 0],
                    [0, 0, 0, 2, 2, 4.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4.5, 2],
                    [0, 0, 0, 0, 0, 0, 2, 4.5]])


def accuracy(labels, pred):
    m = metrics.confusion_matrix(labels, pred)
    i = [*permutations(np.arange(m.shape[1]))]
    all_perm = m[np.arange(m.shape[0]), i]
    return np.max(np.sum(all_perm, 1)) / len(labels)


def common_cov_matrix(cov: np.array, n_rows: list):
    mu_1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
    mu_2 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    mu_3 = np.array([5, 6, 7, 8, 9, 10, 11, 12])

    X1 = np.random.multivariate_normal(mean=mu_1, cov=cov, size=n_rows[0])
    X2 = np.random.multivariate_normal(mean=mu_2, cov=cov, size=n_rows[1])
    X3 = np.random.multivariate_normal(mean=mu_3, cov=cov, size=n_rows[2])

    y = np.zeros(sum(n_rows), dtype="float")
    y[n_rows[0]:n_rows[0] + n_rows[1]] = 1
    y[n_rows[0] + n_rows[1]::] = 2

    X = np.concatenate((X1, X2, X3), axis=0)
    return X, cov, y


if __name__ == "__main__":
    N = 50
    cov = COV_NEGATIVE
    exp_name = "mult1_cov_negative"
    n_rows_data = [500, 300, 200]

    result1 = {"recovered_cov": [], "n_iterations": [], "clustering_accuracy": []}
    result2 = {"recovered_cov": [], "n_iterations": [], "clustering_accuracy": []}
    for exp_i in range(N):
        print(f"\r{exp_i}/{N}", end="")
        X, cov, y_true = common_cov_matrix(cov, n_rows_data)
        em = HierarcicalBiclustering(n_clusters=3, linkage="average", group_search_rng=[2, 3, 4, 5, 6])
        it1 = em.fit(X)

        result1["recovered_cov"].append(em.get_averge_cov())
        result1["n_iterations"].append(it1)
        result1["clustering_accuracy"].append(accuracy(y_true, em.z.argmax(axis=0)))

        em2 = FactorAnalyzerBiclustering(n_clusters=3, q=3)
        it2 = em2.fit(X)
        result2["recovered_cov"].append(em2.get_averge_cov())
        result2["n_iterations"].append(it2)
        result2["clustering_accuracy"].append(accuracy(y_true, em2.z.argmax(axis=0)))

    pd.DataFrame(result1).to_csv(f"mult_components_experiments/{exp_name}_{N}_proposed.csv")
    pd.DataFrame(result2).to_csv(f"mult_components_experiments/{exp_name}_{N}_sanjeena.csv")
    print("\n")
    mape1 = np.abs(np.array(result1["recovered_cov"]) - cov).mean() / np.abs(cov).mean()
    mape2 = np.abs(np.array(result2["recovered_cov"]) - cov).mean() / np.abs(cov).mean()
    print(round(mape1, 3) * 100, round(mape2, 3) * 100)




