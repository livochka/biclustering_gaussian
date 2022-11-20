import numpy as np
from scipy.stats import multivariate_normal


def generate_random_dataset(sizes):
    rows = np.sum(sizes[:, 0])
    columns = np.sum(sizes[:, 1])

    data = np.zeros((rows, columns))
    print(rows, columns)
    row_index = 0
    col_index = 0
    for cluster in range(len(sizes)):
        n_cols = sizes[cluster, 1]
        n_rows = sizes[cluster, 0]

        high = np.random.uniform(1, 3)
        mu_k = np.random.uniform(0, high, n_cols)
        print(mu_k)
        A = np.random.rand(n_cols, n_cols)
        cov_k = A @ A.T
        data[row_index: row_index + n_rows, col_index: col_index + n_cols] = np.random.multivariate_normal(mu_k,
                                                                                                           cov_k,
                                                                                                           n_rows)
        row_index += n_rows
        col_index += n_cols

    zero_rows, zero_columns = np.where(data == 0)
    for i in range(len(zero_rows)):
        row_k, col_k = zero_rows[i], zero_columns[i]
        data[row_k, col_k] = np.random.uniform(-1, 1)

    return data


def common_cov_matrix():
    mu_1 = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
    mu_2 = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    mu_3 = np.array([5, 6, 7, 8, 9, 10, 11, 12])

    cov = np.array([[4.5, 2, 2, 0, 0, 0, 0, 0],
                    [2, 4.5, 2, 0, 0, 0, 0, 0],
                    [2, 2, 4.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4.5, 2, 2, 0, 0],
                    [0, 0, 0, 2, 4.5, 2, 0, 0],
                    [0, 0, 0, 2, 2, 4.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4.5, 2],
                    [0, 0, 0, 0, 0, 0, 2, 4.5]])

    X1 = np.random.multivariate_normal(mean=mu_1, cov=cov, size=500)
    X2 = np.random.multivariate_normal(mean=mu_2, cov=cov, size=300)
    X3 = np.random.multivariate_normal(mean=mu_3, cov=cov, size=200)

    X = np.concatenate((X1, X2, X3), axis=0)
    return X, cov


def generate_one_component_simple():
    mu = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    cov = np.array([[4.5, 2, 2, 0, 0, 0, 0, 0],
                    [2, 4.5, 2, 0, 0, 0, 0, 0],
                    [2, 2, 4.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4.5, 2, 2, 0, 0],
                    [0, 0, 0, 2, 4.5, 2, 0, 0],
                    [0, 0, 0, 2, 2, 4.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 4.5, 2],
                    [0, 0, 0, 0, 0, 0, 2, 4.5]])
    X = np.random.multivariate_normal(mean=mu, cov=cov, size=500)
    return X, cov


def generate_one_component_negative_covariance():
    mu = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    cov = np.array([[4.5, -2, 1, 0, 0, 0, 0, 0],
                    [-2, 4.5, 2, 0, 0, 0, 0, 0],
                    [1, 2, 4.5, 0, 0, 0, 0, 0],
                    [0, 0, 0, 4.5, -2, 2, 0, 0],
                    [0, 0, 0, -2, 4.5, 2, 0, 0],
                    [0, 0, 0, 2, 2, 4.5, 0, 0],
                    [0, 0, 0, 0, 0, 0, 3, 2],
                    [0, 0, 0, 0, 0, 0, 2, 4.5]])
    X = np.random.multivariate_normal(mean=mu, cov=cov, size=500)
    return X, cov


def get_true_groups1():
    groups = []

    for i in range(3):
        groups.append(0)

    for i in range(3, 6):
       groups.append(1)

    for i in range(6, 8):
        groups.append(2)

    return np.array(groups)




if __name__ == "__main__":
    # sizes = np.array([[5, 10], [4, 8]])
    # data = generate_random_dataset(sizes)
    # print(data.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(data, )
    # plt.show()
   common_cov_matrix()

