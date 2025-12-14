import numpy as np

class MC2PCA:
    def __init__(self, p, tol=1e-7, max_iter=100):
        self.p = p
        self.tol = tol
        self.max_iter = max_iter

    def _normalize_all(self, X):
        normalized = []
        for Xi in X:
            Xi_c = Xi - Xi.mean(axis=0)
            normalized.append(Xi_c)
        return normalized

    def _compute_covariances(self, X):
        covs = []
        for Xi in X:
            covs.append(np.cov(Xi.T, bias=True))
        return covs

    def _CPCA(self, covs_cluster):
        mean_cov = np.mean(covs_cluster, axis=0)
        _, val_propre, vt = np.linalg.svd(mean_cov)
        val_propre = val_propre**2             
        var_ratio = np.sum(val_propre[:self.p]) / np.sum(val_propre)
        return vt[:self.p, :].T, var_ratio

    def _compute_CPCA_for_all_clusters(self, covs, clusters):
        S = []
        info = []
        for idx in clusters:
            if len(idx) > 0:
                Sk, vk = self._CPCA([covs[i] for i in idx])
                S.append(Sk)
                info.append(vk)
            else:
                S.append(None)
                info.append(0.0)
        return S, info

    def _assign_clusters(self, X, S):
        n = len(X)
        K = len(S)
        Error = np.zeros((n, K))

        for k in range(K):
            if S[k] is not None:
                sst = S[k] @ S[k].T
                for i in range(n):
                    time_series = X[i]                 
                    Y = time_series @ sst
                    err = np.linalg.norm(time_series - Y, axis=1)
                    Error[i, k] = np.mean(err)
            else:
                Error[:, k] = np.inf

        I = np.argmin(Error, axis=1)
        v = Error[np.arange(n), I]
        return I, v, Error

    def fit(self, series_list, K):
        X = self._normalize_all(series_list)
        covs = self._compute_covariances(X)

        n = len(X)
        idx = np.array_split(np.arange(n), K)
        idx = [list(c) for c in idx]

        S, _ = self._compute_CPCA_for_all_clusters(covs, idx)

        E = [np.inf]

        for _ in range(self.max_iter):
            I, v, _ = self._assign_clusters(X, S)
            E.append(np.sum(v) / len(v))

            if abs(E[-2] - E[-1]) < self.tol:
                break

            idx = [np.where(I == k)[0].tolist() for k in range(K)]
            S, info = self._compute_CPCA_for_all_clusters(covs, idx)

        self.assignments = I
        self.clusters = idx
        self.subspaces = S
        self.variances = info
        self.errors = E

        return self
