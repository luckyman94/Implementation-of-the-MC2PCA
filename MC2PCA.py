import numpy as np

class MC2PCA:
    def __init__(self, p, tol, max_iter):
        self.p = p
        self.tol = tol
        self.max_iter = max_iter

    def _normalize_all(self, X):
        normalized = []
        for i in X:
            X_norm = i - i.mean(axis=0)    
            normalized.append(X_norm)
        return normalized
    
    def _compute_covariances(self, X):
        covs = []
        for i in X:
            sigma = np.cov(i.T)
            covs.append(sigma)
        return covs

    def _CPCA(self, covs_cluster):
        cov_mean = np.mean(covs_cluster, axis=0)
        _, S, Vt = np.linalg.svd(cov_mean)
        S = S**2   
        var_ratio = S[:self.p].sum() / S.sum()
        return Vt[:self.p].T, var_ratio


    def _compute_CPCA_for_all_clusters(self, covs, clusters):
        subspaces = []
        variances = []

        for idx_list in clusters:
            if len(idx_list) == 0:
                subspaces.append(None)
                variances.append(0.0)
                continue

            covs_cluster = [covs[j] for j in idx_list]

            S_k, var_k = self._CPCA(covs_cluster)

            subspaces.append(S_k)
            variances.append(var_k)

        return subspaces, variances
    
    def _assign_clusters(self, series, subspaces):
        n = len(series)
        K = len(subspaces)
        error_matrix = np.zeros((n, K))

        for i, Xi in enumerate(series):
            # Xi : T x d
            for k in range(K):
                S_k = subspaces[k]
                if S_k is None:
                    error_matrix[i,k] = np.inf
                    continue
                
                P_k = S_k @ S_k.T
                Xi_hat = Xi @ P_k

                # Erreur dimension par dimension (comme GitHub)
                per_dim_errors = np.linalg.norm(Xi - Xi_hat, axis=0)
                error_matrix[i,k] = per_dim_errors.mean()

        assignments = np.argmin(error_matrix, axis=1)
        min_errors = error_matrix[np.arange(n), assignments]

        return assignments, np.sum(min_errors), np.mean(min_errors), error_matrix


    

    def fit(self, series_list, K):
        normalized = self._normalize_all(series_list)
        covs = self._compute_covariances(normalized)

        n = len(series_list)

        clusters = np.array_split(np.arange(n), K)
        clusters = [c.tolist() for c in clusters]

        prev_error = np.inf
        errors = []

        for i in range(self.max_iter):
            subspaces, variances = self._compute_CPCA_for_all_clusters(covs, clusters)

            assignments, total_error, mean_error, _ = self._assign_clusters(normalized, subspaces)
            errors.append(mean_error)          


            if abs(prev_error - total_error) < self.tol:
                break

            prev_error = total_error
            clusters = [np.where(assignments == k)[0].tolist() for k in range(K)]


        self.assignments = assignments
        self.clusters = clusters
        self.subspaces = subspaces
        self.variances = variances
        self.errors = errors

        return self
