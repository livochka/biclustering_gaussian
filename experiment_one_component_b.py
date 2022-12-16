from utils.group_estimators import GroupEstimatorNumerical, GroupEstimatorHierarchical, GroupEstimatorGreedy
import numpy as np
import timeit
import pandas as pd
import pickle as pkl
from sklearn.metrics import silhouette_score


def generate_random_params(n_columns, columns_per_block):
    mu = np.random.random(n_columns)
    cov = np.zeros((n_columns, n_columns))
    start, end = 0, columns_per_block
    block_index = 0
    true_groups = np.zeros(n_columns)
    while start < n_columns:
        A = np.random.random((columns_per_block, columns_per_block)) + 1
        sub_cov = A.T @ A
        cov[start:end, start:end] = sub_cov
        true_groups[start:end] = block_index

        start, end = end, end + columns_per_block
        block_index += 1

    B = np.random.random((n_columns, n_columns))
    cov += B.T @ B * 0.5

    return mu, cov, true_groups.astype("int")


def generate_and_save(ncols: list, columns_per_block: list, names: list):
    for i in range(len(ncols)):
        mu, cov, true_groups = generate_random_params(ncols[i], columns_per_block[i])

        with open(f"experiment_b_cov_matrices/{names[i]}.pkl", "wb") as f:
            pkl.dump({"mu": mu,
                      "cov": cov,
                      "true_groups": true_groups}, f)


def pick_n_clusters(data, estimator_class, n_clusters):
    best_score = -1
    best_cl = 0
    cov = np.cov(data.T)
    for n in n_clusters:
        if n > 1:
            estimator = estimator_class(n_clusters=n)
            estimator.fit(data)
            score_curr = silhouette_score(cov, estimator.predicted_groups)
            if score_curr > best_score:
                best_cl = n
                best_score = score_curr
    return best_cl


if __name__ == "__main__":
    cols = [24, 24, 24, 24 * 4, 24 * 4, 24*4, 24*16, 24*16, 24*16]
    cols_per_block = [8, 6, 3, 8 * 2, 6*2, 3*2,8 * 4, 6*4, 3*4]
    names = ["low_low", "low_medium", "low_high", "medium_low", "medium_medium", "medium_high",
             "high_low", "high_medium", "high_high"]

    #generate_and_save(cols, cols_per_block, names)
    result = {"name": [], "n_correct": [], "N": [], "ncols": [], "n_blocks": [],#}
              # "n_cluster_correct": [],
              # "cluster_search_range": []
              }
    N = 500
    for i in range(len(cols)):
        correct = 0
        correct_n_clusters = 0

        true_cl = cols[i] // cols_per_block[i]
        rng = int(np.floor(np.sqrt(true_cl)) * 2)

        for n in range(N):
            mu, cov, true_groups = generate_random_params(cols[i], cols_per_block[i])
            X = np.random.multivariate_normal(mean=mu, cov=cov, size=50)#cols[i] // 2)
            possible_clusters = [true_cl - j for j in range(rng, -rng, -1)]
            n_cl = pick_n_clusters(X, GroupEstimatorHierarchical, possible_clusters)
            correct_n_clusters += n_cl == true_cl
           # estimator = GroupEstimatorHierarchical(n_clusters=true_cl)
            estimator = GroupEstimatorHierarchical(n_clusters=n_cl,
                                                   metric="precomputed"
                                                   )

            estimator.fit(X)
            correct += estimator.is_correct(true_groups)

        print(f"{names[i]}: {round(correct / N, 3) * 100} [{correct}/{N}], {correct_n_clusters}")
        result["name"].append(names[i])
        result["n_correct"].append(correct)
        result["N"].append(N)
        result["ncols"].append(cols[i])
        result["n_blocks"].append(cols[i] // cols_per_block[i])
        # result["n_cluster_correct"].append(correct_n_clusters)
        # result["cluster_search_range"].append(rng)


    #pd.DataFrame(result).to_csv("experiment_b_results/accuracy_by_columns_blocks_KNOWN_n_clusters.csv")




