import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA



class ClusteringMetrics:
    def __init__(self):
        self.results = []  

    def add_run(self, y_true, y_pred, p=None, X=None, errors=None, noise=None):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        assert len(y_true) == len(y_pred)

        entry = {
            "p": p,
            "ARI": adjusted_rand_score(y_true, y_pred),
            "NMI": normalized_mutual_info_score(y_true, y_pred),
            "Purity": self._purity(y_true, y_pred),
            "Precision": self._clustering_precision(y_true, y_pred),
            "Recall": self._clustering_recall(y_true, y_pred),
            "y_true": y_true,
            "y_pred": y_pred,
            "X": X,
            "errors": errors,
            "noise": noise
}

        self.results.append(entry)



    def _purity(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        return np.sum(np.max(cm, axis=0)) / np.sum(cm)

    def summary(self):
        for r in self.results:
            print(
                f"p={r['p']} | "
                f"ARI={r['ARI']:.3f} | "
                f"NMI={r['NMI']:.3f} | "
                f"Precision={r['Precision']:.3f} | "
                f"Recall={r['Recall']:.3f}"
            )


    def plot_metrics_vs_p(self):
        ps = [r["p"] for r in self.results]

        plt.figure(figsize=(7,5))
        for metric in ["ARI", "NMI", "Precision"]:
            values = [r[metric] for r in self.results]
            plt.plot(ps, values, marker="o", label=metric)

        plt.xlabel("p (CPCA dimension)")
        plt.ylabel("Score")
        plt.title("Clustering performance vs p")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_metrics_mean_std(self):
        import pandas as pd

        df = pd.DataFrame(self.results)
        grouped = df.groupby("p")[["ARI", "NMI", "Precision"]].agg(["mean", "std"])

        plt.figure(figsize=(7,5))
        for metric in ["ARI", "NMI", "Precision"]:
            mean = grouped[(metric, "mean")]
            std = grouped[(metric, "std")]
            plt.errorbar(mean.index, mean, yerr=std, marker="o", label=metric)

        plt.xlabel("p")
        plt.ylabel("Score")
        plt.title("MC2PCA metrics (mean ± std)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_convergence_all_runs(self, p=None):
        plt.figure(figsize=(7,5))

        for r in self.results:
            if p is not None and r["p"] != p:
                continue
            if r["errors"] is None:
                continue
            plt.plot(r["errors"], alpha=0.7)

        plt.xlabel("Iteration")
        plt.ylabel("Mean reconstruction error")
        plt.title("MC2PCA convergence under different initializations")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_convergence_mean_std(self, p=None):
        all_errors = []

        for r in self.results:
            if p is not None and r["p"] != p:
                continue
            if r["errors"] is not None:
                all_errors.append(np.array(r["errors"]))

        if len(all_errors) == 0:
            return

        max_len = max(len(e) for e in all_errors)
        mat = np.zeros((len(all_errors), max_len))

        for i, e in enumerate(all_errors):
            mat[i, :len(e)] = e
            mat[i, len(e):] = e[-1]

        mean = mat.mean(axis=0)
        std = mat.std(axis=0)

        plt.figure(figsize=(7,5))
        plt.plot(mean, label="Mean")
        plt.fill_between(
            np.arange(len(mean)),
            mean - std,
            mean + std,
            alpha=0.3,
            label="± std"
        )

        plt.xlabel("Iteration")
        plt.ylabel("Mean reconstruction error")
        plt.title("MC2PCA convergence (mean ± std)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def convergence_stats(self, p=None):
        finals = []
        n_iters = []

        for r in self.results:
            if p is not None and r["p"] != p:
                continue
            if r["errors"] is not None:
                finals.append(r["errors"][-1])
                n_iters.append(len(r["errors"]))

        print("Convergence statistics")
        print("----------------------")
        print(f"Final error : mean={np.mean(finals):.4f} ± {np.std(finals):.4f}")
        print(f"Iterations : mean={np.mean(n_iters):.1f}")


    
    def plot_cluster_size_boxplot(self, p=None):
        

        cluster_sizes_per_run = []

        for r in self.results:
            if p is not None and r["p"] != p:
                continue

            y_pred = r["y_pred"]
            labels, counts = np.unique(y_pred, return_counts=True)
            cluster_sizes_per_run.append(dict(zip(labels, counts)))

        if len(cluster_sizes_per_run) == 0:
            print("No runs available.")
            return

        all_clusters = sorted(
            set(k for d in cluster_sizes_per_run for k in d.keys())
        )

        data = []
        for k in all_clusters:
            sizes_k = [d.get(k, 0) for d in cluster_sizes_per_run]
            data.append(sizes_k)

        plt.figure(figsize=(8,5))
        plt.boxplot(data, labels=[f"C{k}" for k in all_clusters], showfliers=True)

        plt.xlabel("Cluster")
        plt.ylabel("Cluster size")
        plt.title("Cluster size distribution across initializations")
        plt.grid(axis="y")
        plt.tight_layout()
        plt.show()


    def _clustering_precision(self, y_true, y_pred):
    
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        N = len(y_true)
        precision = 0.0

        pred_clusters = {
            k: set(np.where(y_pred == k)[0])
            for k in np.unique(y_pred)
        }
        true_groups = {
            k: set(np.where(y_true == k)[0])
            for k in np.unique(y_true)
        }

        for cj in pred_clusters.values():
            if len(cj) == 0:
                continue
            max_inter = max(len(cj & gi) for gi in true_groups.values())
            precision += (len(cj) / N) * (max_inter / len(cj))

        return precision


    def _clustering_recall(self, y_true, y_pred):
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        N = len(y_true)
        recall = 0.0

        pred_clusters = {
            k: set(np.where(y_pred == k)[0])
            for k in np.unique(y_pred)
        }
        true_groups = {
            k: set(np.where(y_true == k)[0])
            for k in np.unique(y_true)
        }

        for gi in true_groups.values():
            if len(gi) == 0:
                continue
            max_inter = max(len(gi & cj) for cj in pred_clusters.values())
            recall += (len(gi) / N) * (max_inter / len(gi))

        return recall



