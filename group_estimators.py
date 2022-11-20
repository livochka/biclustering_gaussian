import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import AgglomerativeClustering

from data_generation import generate_one_component_negative_covariance, generate_one_component_simple, get_true_groups1
from parameter_initialization import ParamsInitializer



class GroupEstimator(ABC):

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.predicted_groups = None

    @abstractmethod
    def fit(self, data):
        pass

    @staticmethod
    def mu_cov_MLE(data):
        mu = np.mean(data, axis=0)
        cov = np.cov(data.T)
        return mu, cov

    def is_correct(self, true_groups):
        comb_result = set()
        comb_true = set()

        for group_i in range(self.predicted_groups.max() + 1):
            features_group = np.where(self.predicted_groups == group_i)[0]
            comb_result.add(tuple(features_group))

            features_group_true = np.where(true_groups == group_i)[0]
            comb_true.add(tuple(features_group_true))

        return comb_result == comb_true

    @staticmethod
    def compute_likelihood(mu, cov_constrained, data):
        L = 0
        # cov_constrained = np.zeros(cov.shape, dtype=float)
        #
        # for j in range(len(S)):
        #     cov_constrained = cov_constrained + S[j] @ cov @ S[j]
        #

        cov_constrained_inv = np.linalg.inv(cov_constrained)

        for j in range(data.shape[0]):
            x = data[j, :]
            l_row = -0.5 * np.log(np.linalg.det(cov_constrained)) - 0.5 * (x - mu).T @ cov_constrained_inv @ (
                        x - mu)
            L += l_row
        return L


class GroupEstimatorHierarchical(GroupEstimator):

    def __init__(self, n_clusters, linkage="average"):
        self.linkage = linkage
        super().__init__(n_clusters)

    @staticmethod
    def compute_corr_matrix(cov):
        sdr = np.sqrt(np.diag(cov))
        corr_matrix = np.diag(1 / sdr) @ cov @ np.diag(1 / sdr)
        return corr_matrix

    def fit(self, data):
        mu, cov = self.mu_cov_MLE(data)
        corr_matrix = self.compute_corr_matrix(cov)
        model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        model.fit(np.abs(corr_matrix))

        self.predicted_groups = model.labels_


class GroupEstimatorNumerical(GroupEstimator):

    def __init__(self, n_clusters, gamma=0.001, lambd=0.0001, step=0.1, conv_epsilon=0.1):
        self.gamma = gamma
        self.lambd = lambd
        self.step = step

        # used for iterative optimization
        self.LL = None
        self.LL_inf = None
        self.epsilon = conv_epsilon

        super().__init__(n_clusters)

    def initialize_D(self, data):
        initialization = ParamsInitializer(n_clusters=1)
        z = initialization.intialize_z(data, "kmeans")
        D = initialization.initialize_S(data, z, n_groups=self.n_clusters)[0]
        return D

    @staticmethod
    def computeT(D, R, cov):
        result = np.zeros(cov.shape, dtype="float")
        R_inv = np.linalg.inv(R)
        for D_i in D:
            result += D_i @ cov @ D_i
        return np.linalg.inv(result)

    def check_aitkens_criterion(self, likelihood):
        self.LL.append(likelihood)

        if len(self.LL) > 2:
            alpha = (self.LL[-1] - self.LL[-2]) / (self.LL[-2] - self.LL[-3])
            LL_inf_curr = self.LL[-2] + (self.LL[-1] - self.LL[-2]) / (1 - alpha)
            self.LL_inf.append(LL_inf_curr)

            if len(self.LL_inf) > 1:
                # print("\t", np.abs(self.LL_inf[-1] - self.LL_inf[-2]))
                return np.abs(self.LL_inf[-1] - self.LL_inf[-2]) < self.epsilon

        return False

    # def check_simple_criterion(self, likelihood):
    #     self.LL.append(likelihood)
    #
    #     if len(self.LL) > 1:
    #         return np.abs(self.LL[-1] - self.LL[-2]) <

    def gradient_opt(self, data, cov, mu, R, T, D, lambd, gamma):
        num_s = len(D)
        s = np.array([np.diag(D_j) for D_j in D])
        #S = s.T

        def gradient(T, R, S, gamma, lambd, s_j, j):
            R_inv = np.linalg.inv(R)
            return (T * R_inv) @ s_j - (R_inv * R) @ s_j + gamma * s_j - lambd * 2 * (S @ np.linalg.inv(S.T @ S))[:, j]

        def updateT(s, R):
            D = [np.diag(s_i) for s_i in s]
            return self.computeT(D, R, cov)

        self.LL = []
        self.LL_inf = []

        stop = False
        it = 0
        #for iter in range(1000):
        while not stop:
           # print(f"IT = {it}")
            s_all = 0
            for i in range(num_s):
                s_next = s[i, :] + gradient(T, R, s.T, gamma, lambd, s[i, :], i) * self.step
                s_all += s_next
                s[i, :] = s_next

            T = updateT(s, R)
          #  likelihood = self.compute_likelihood(mu, T, data)
            stop = it >= 1000
            it += 1

          #  S = np.array(s).T
        return s

    def fit(self, data):
        mu, cov = self.mu_cov_MLE(data)
        D = self.initialize_D(data)
        R = np.linalg.inv(cov)
        T = self.computeT(D, R, cov)
        result = self.gradient_opt(data, cov, mu, R, T, D, self.lambd, self.gamma)
        self.predicted_groups = np.argmax(result, axis=0)


if __name__ == "__main__":
    # for negative covariance: gamma=0.1, lambd=0.0001
    # for positive covariance:gamma=0.1, lambd=0.0001
    estimator1 = GroupEstimatorNumerical(n_clusters=3, gamma=0.001, lambd=0.0001, conv_epsilon=0.1, step=0.1)
    estimator2 = GroupEstimatorHierarchical(n_clusters=3)
    correct_groups = get_true_groups1()

    N = 50
    correct_1 = 0
    correct_2 = 0

    for exp in range(N):
        data, true_cov = generate_one_component_simple()
        estimator1.fit(data)
        correct_1 += estimator1.is_correct(correct_groups)

        estimator2.fit(data)
        correct_2 += estimator2.is_correct(correct_groups)

    print(correct_1 / N, correct_2 / N)

        # print(estimator.predicted_groups)













