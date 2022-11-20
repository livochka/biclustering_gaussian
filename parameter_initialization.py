from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy as np


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
            print(f"cluster {cluster} data shape:", sample.shape)
            S_k = np.cov(sample.T)
            S.append(S_k)
            print(f"cluster {cluster} cov shape:", S_k.shape, "\n")

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















