"""
This module implements hierarchical and factor-analyzer bi-clustering methods
"""
import numpy as np
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np
from sklearn import metrics
from itertools import permutations


class ParamsInitializer:

    def __init__(self, n_clusters):
        self.n = n_clusters

    def intialize_z(self, data, method="kmeans"):
        if method == "kmeans":
            model = KMeans(n_clusters=self.n).fit(data)
            z = []
            for i in range(self.n):
                z.append(np.array([int(j == i) for j in model.labels_]))
            return np.array(z)
        else:
            raise NotImplementedError(f"Initialization method {method} is not implemented!")

    def initialize_DTS(self, data, z, q):
        D = []
        T = []
        S = []

        for cluster in range(self.n):
            z_k = z[cluster]
            sample = data[z_k == 1]
            if sample.shape[0] < 2:
                add_point_ind = int(np.random.randint(data.shape[0]))
                sample = np.concatenate([sample, data[add_point_ind, :].reshape(1, -1)], axis=0)
            S_k = np.cov(sample.T)
            S.append(S_k)

            pca_model = PCA(n_components=q)
            pca_model.fit(S_k)
            sv = pca_model.singular_values_

            T_k = np.diag(sv)
            T.append(T_k)

            V = pca_model.components_.T
            D_k = np.diag(np.diag(S_k - V @ T_k @ V.T))
            D.append(D_k)

        return D, T, S

    def initialize_B(self, data, z, q):
        data = scale(data)
        B = []

        for cluster in range(self.n):
            z_k = z[cluster]
            sample = data[z_k == 1]
            S_k = np.cov(sample.T)

            pca_model = PCA(n_components=q)
            pca_model.fit(S_k)
            V = pca_model.components_.T
            L_k = (np.max(V, axis=1, keepdims=True) == V) * 1
            B.append(L_k)

        return B

    def initialize_S(self, data, z, n_groups):
        data = scale(data)
        S = []
        for cluster in range(self.n):
            z_k = z[cluster]
            sample = data[z_k == 1]
            S_k = np.cov(sample.T)
            pca_model = PCA(n_components=n_groups)
            pca_model.fit(S_k)

            V = pca_model.components_.T
            L_k = (np.max(V, axis=1, keepdims=True) == V) * 1

            S_cluster = [np.zeros(S_k.shape, dtype=float), np.zeros(S_k.shape, dtype=float), np.zeros(S_k.shape, dtype=float)]
            for i in range(L_k.shape[1]):
                for j in range(L_k.shape[0]):
                    if L_k[j, i] == 1:
                        S_cluster[i][j, j] = 1

            S_cluster = np.array(S_cluster)
            S.append(S_cluster)

        return np.array(S)

class AbstractEM:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.initializer = ParamsInitializer(n_clusters)
        self.mu = np.array([None in range(n_clusters)])
        self.pi = np.array([None in range(n_clusters)])
        self.z = np.array([None in range(n_clusters)])
        self.n = np.array([None in range(n_clusters)])
        self.L = []
        self.L_inf = []

    @abstractmethod
    def fit(self, data):
        pass

    def check_convergence(self, eps=0.01):
        if len(self.L) > 2:
            alpha = (self.L[-1] - self.L[-2]) / (self.L[-2] - self.L[-3])
            L_inf = self.L[-2] + (self.L[-1] - self.L[-2]) / (1 - alpha)
            self.L_inf.append(L_inf)
            if len(self.L_inf) > 1:
                div = np.abs(self.L_inf[-1] - self.L_inf[-2])
               # print(f"\tdiv = {div}")
                if div < eps:
                    return True
    
    def fit_predict(self, data):
        self.fit(data)
        return self.get_predicted_labels()
            
    def get_predicted_labels(self):
        return self.z.argmax(axis=0)


class HierarcicalBiclustering(AbstractEM):

    def __init__(self, n_clusters, linkage, group_search_rng, shared_cov=False, metric=None):
        self.cov = np.array([None in range(n_clusters)])
        self.corr = np.array([None in range(n_clusters)])
        self.D = np.array([None in range(n_clusters)])
        self.linkage = linkage
        self.group_search_rng = group_search_rng
        self.shared_cov = shared_cov
        self.metric = metric
        super().__init__(n_clusters)

    def fit(self, data, n_it=20):
        self._initialize_params(data)
        for step in range(n_it):
            # print(f"\nEPOCH = {step}")
            self.m_step(data)
            self.e_step(data)

            # CHECK CRITERIUM
            L = self.loss(data)
            self.L.append(L)

            if self.check_convergence(eps=0.00001):
                break
        return step

    def _initialize_params(self, data, z_method="kmeans"):
        self.z = self.initializer.intialize_z(data, z_method)
        self.n = np.sum(self.z, axis=1)

    @staticmethod
    def compute_corr_matrix(cov):
        sdr = np.sqrt(np.diag(cov))
        corr_matrix = np.diag(1 / sdr) @ cov @ np.diag(1 / sdr)
        return corr_matrix

    @staticmethod
    def compute_cov_constrained(cov, D):
        cov_constrained = np.zeros(cov.shape, dtype=float)

        for j in range(len(D)):
            cov_constrained = cov_constrained + D[j] @ cov @ D[j]
        return cov_constrained

    def _run_agglomerative_clustering(self, corr, n_groups):
        model = AgglomerativeClustering(n_clusters=n_groups, linkage=self.linkage, metric=self.metric)
        R = np.abs(corr)
        if self.metric:
            R = 1 - R
        model.fit(R)
        return model.labels_

    def _get_labels_best_n_groups(self, corr):
        best_score = -1
        best_cl, best_labels = 0, None
        for k in self.group_search_rng:
            labels = self._run_agglomerative_clustering(corr, k)
            score_curr = silhouette_score(corr, labels)
            if score_curr > best_score:
                best_cl = k
                best_score = score_curr
                best_labels = labels
        return best_labels, best_cl

    def estimate_D(self, corr):
        labels, n_groups = self._get_labels_best_n_groups(corr)
       # print(f"best labels = {labels}, best n groups = {n_groups}")
        D = []
        for i in range(n_groups):
            D_k = np.zeros(corr.shape)
            group_indx = np.where(labels == i)[0]
            D_k[group_indx, group_indx] = 1
            D.append(D_k)
        return np.array(D)

    def e_step(self, data):
        z = np.zeros((self.n_clusters, data.shape[0]), dtype=float)
        log_probs = []
        for i in range(self.n_clusters):
            cov = self._get_cov_matrix_cluster(i)
            p = multivariate_normal.logpdf(data, self.mu[i], cov)
            log_probs.append(p + np.log(self.pi[i]))
        
        log_probs = np.array(log_probs)
        max_log_probs = np.max(log_probs, axis=0)

        log_probs -= max_log_probs
        probs = np.exp(log_probs)


        denom = probs.sum(axis=0)
        for i in range(self.n_clusters):
            z[i, :] = probs[i] / denom

        self.z = z
        self.n = np.sum(self.z, axis=1)

    def m_step(self, data):
        self.pi = self.n / data.shape[0]

        # ESTIMATING mu_k
        mu = []
        for i in range(self.n_clusters):
            mu_k = (data * self.z[i].reshape([-1, 1])).sum(axis=0) / self.n[i]
            mu.append(mu_k)

        self.mu = np.array(mu)

        cov = []
        reg_cov = 1e-6 * np.eye(data.shape[1])

        # Estimating Sigma_k
        for i in range(self.n_clusters):
            cov_k = self.z[i] * (data - self.mu[i]).T @ (data - self.mu[i]) / self.n[i] + reg_cov
            cov.append(cov_k)
        cov = np.array(cov)

        if self.shared_cov:
            cov_shared = np.average(cov, weights=self.pi, axis=0)
            corr = self.compute_corr_matrix(cov_shared)
            D = self.estimate_D(corr)

            self.D = D
            self.cov = self.compute_cov_constrained(cov_shared, D)
        else:
            D = []
            for i in range(self.n_clusters):
                corr = self.compute_corr_matrix(cov[i])
                D_k = self.estimate_D(corr)
                D.append(D_k)
                cov[i, :, :] = self.compute_cov_constrained(cov[i], D_k)

            self.D = D
            self.cov = cov

    def loss(self, data):
        L = 0
        for j in range(data.shape[0]):
            row = data[j, :]
            l_row = 0
            for i in range(self.n_clusters):
                cov = self._get_cov_matrix_cluster(i)
                l_row += self.pi[i] * multivariate_normal.pdf(row, self.mu[i], cov)
            L += np.log(l_row)
        return L

    def get_averge_cov(self):
        return self.cov.mean(axis=0)

    def _get_cov_matrix_cluster(self, cluster_ind):
        if self.shared_cov:
            return self.cov
        return self.cov[cluster_ind]

    def _get_D_cluster(self, cluster_ind):
        if self.shared_cov:
            return self.D
        return self.D[cluster_ind]


class FactorAnalyzerBiclustering(AbstractEM):

    def __init__(self, n_clusters, q):
        self.q = q
        self.D = np.array([None in range(n_clusters)])
        self.T = np.array([None in range(n_clusters)])
        self.B = np.array([None in range(n_clusters)])
        self.S = np.array([None in range(n_clusters)])

        super().__init__(n_clusters)


    def fit(self, data):
        self._initialize_params(data, z_method="kmeans")

        for step in range(100):
            self.first_cycle_e_step(data)
            self.first_cycle_cm_step(data)
            U = self.second_cycle_e_step(data)
            self.second_cycle_cm_step(data, U)

            # CHECK CRITERIUM
            L = self.loss(data)
            self.L.append(L)

            if self.check_convergence():
                break
        return step

    def _initialize_params(self, data, z_method):
        self.z = self.initializer.intialize_z(data, z_method)
        self.n = np.sum(self.z, axis=1)
        reg_cov = 1e-6 * np.eye(data.shape[1])

        D, T, S = self.initializer.initialize_DTS(data, self.z, self.q)
        self.D = D + reg_cov
        self.T = T
        self.S = S

        B = self.initializer.initialize_B(data, self.z, self.q)
        self.B = B

        self.first_cycle_cm_step(data)

    def first_cycle_e_step(self, data):
        self.update_z(data)

    def update_z(self, data):

        z = np.zeros((self.n_clusters, data.shape[0]), dtype=float)
        log_probs = []
        for i in range(self.n_clusters):
            cov_k = self.B[i] @ self.T[i] @ self.B[i].T + self.D[i]
            p = multivariate_normal.logpdf(data, self.mu[i], cov_k)
            log_probs.append(p + np.log(self.pi[i]))
        
        log_probs = np.array(log_probs)
        max_log_probs = np.max(log_probs, axis=0)

        log_probs -= max_log_probs
        probs = np.exp(log_probs)


        denom = probs.sum(axis=0)
        for i in range(self.n_clusters):
            z[i, :] = probs[i] / denom

        # updating self.z
        self.z = z
        self.n = np.sum(self.z, axis=1)

    def first_cycle_cm_step(self, data):
        self.pi = self.n / data.shape[0]
        mu = []
        for i in range(self.n_clusters):
            mu_k = (data * self.z[i].reshape([-1, 1])).sum(axis=0) / self.n[i]
            mu.append(mu_k)
        self.mu = np.array(mu)

    def second_cycle_e_step(self, data):
        # update z_ik and u_ik
        self.update_z(data)

        U = []
        for i in range(self.n_clusters):
            A_k = self.T[i] @ self.B[i].T @ np.linalg.inv(self.B[i] @ self.T[i] @ self.B[i].T + self.D[i])
            U_k = A_k @ (data - self.mu[i]).T
            U.append(U_k)

        return np.array(U)

    def second_cycle_cm_step(self, data, U):
        # update S_k, D_k, T_k, B_k
        reg_cov = 1e-6 * np.eye(data.shape[1])

        # computing T & D
        T = []
        D = []
        S = []
        B = []
        for i in range(self.n_clusters):
            cov_k_inv = np.linalg.inv(self.B[i] @ self.T[i] @ self.B[i].T + self.D[i])
            latent = (self.z[i].reshape(1, -1) * U[i]) @ U[i].T / self.n[i]
            theta_k = self.T[i] - self.T[i] @ self.B[i].T @ cov_k_inv @ self.B[i] @ self.T[i] + latent
            S_next = self.z[i] * (data - self.mu[i]).T @ (data - self.mu[i]) / self.n[i]
            T_next = np.diag(np.diag(theta_k))

            S.append(S_next)
            T.append(T_next)

            cov_k_inv_next = np.linalg.inv(self.B[i] @ T_next @ self.B[i].T + self.D[i])
            A_k = S_next - 2 * self.B[i] @ T_next @ self.B[i].T @ cov_k_inv_next @ S_next + self.B[i] @ theta_k @ self.B[i].T
            D_next = np.diag(np.diag(A_k))

            D.append(D_next)

            B_next = self._estimate_B(data, U, S_next, T_next, D_next, theta_k, i)
            B.append(B_next)

        self.T = T
        self.D = D + reg_cov
        self.S = S
        self.B = B

    def _estimate_B(self, data, U, S_k, T_k, D_k, theta_k, k):
        B = self.B[k]

        for row in range(B.shape[0]):
            #print("ROW: ", row)
            max_B = B
            max_q = - 10 ** 6
            for column in range(B.shape[1]):
                row_k = np.zeros(B.shape[1], dtype=float)
                row_k[column] = 1
                B_current = B.copy()
                B_current[row, :] = row_k

                q_current = self.Q2_k(data, U, theta_k, S_k, T_k, D_k, B_current, k)
                #print("\tQ: ", q_current, max_q)
                if q_current >= max_q:
                    max_q = q_current
                    max_B = B_current
            B = max_B

        return B




    def Q2_k(self, data, U, theta_k, S_k, T_k, D_k, B_k, k):
        D_inv = np.linalg.inv(D_k)
        T_inv = np.linalg.inv(T_k)

        Q = np.log(np.linalg.det(D_inv)) + np.log(np.linalg.det(T_inv))
        Q -= np.trace(D_inv @ S_k)
        Q -= np.trace(T_inv @ theta_k)

        sum_k = 0
        for i in range(data.shape[0]):
            sum_k += self.z[k, i] * (data[i, :] - self.mu[k]) @ D_inv @ B_k @ U[k, :, i]
        Q += sum_k

        Q -= np.trace(D_inv @ B_k @ theta_k @ B_k.T)

        return Q

    def l2(self, data, U):
        L = 0
        for k in range(self.n_clusters):
            D_inv = np.linalg.inv(self.D[k])
            T_inv = np.linalg.inv(self.T[k])
            a = np.log(self.pi[k]) + 0.5 * np.log(np.linalg.det(D_inv)) + 0.5 * np.log(np.linalg.det(T_inv))
            L_k = self.z[k].sum() * a
            zUU = self.z[k].reshape(1, -1) * U[k] @ U[k].T
            L_k -= 0.5 * np.trace(T_inv @ zUU)
            L_k -= 0.5 * np.trace(D_inv @ (self.z[k].reshape(1, -1) * (data - self.mu[k]).T @ (data - self.mu[k])))

            sum_k = 0
            for i in range(data.shape[0]):
                sum_k += self.z[k, i] * (data[i, :] - self.mu[k]) @ D_inv @ self.B[k] @ U[k, :, i]

            L_k += sum_k

            L_k -= 0.5 * np.trace(self.B[k].T @ D_inv @ self.B[k] @ zUU)
            L += L_k
        return L

    def loss(self, data):
        L = 0
        for j in range(data.shape[0]):
            row = data[j, :]
            l_row = 0
            for i in range(self.n_clusters):
                cov = self.B[i] @ self.T[i] @ self.B[i].T + self.D[i]
                l_row += self.pi[i] * multivariate_normal.pdf(row, self.mu[i], cov)
            L += np.log(l_row)
        return L

    def get_averge_cov(self):
        cov = []
        for component in range(self.n_clusters):
            cov.append(self.B[component] @ self.T[component] @ self.B[component].T + self.D[component])
        return np.array(cov).mean(axis=0)
