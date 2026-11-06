import numpy as np
import pandas as pd

# ================= HELPER FUNCTIONS (replace sklearn) =================
def make_blobs(n_samples=200, centers=4, cluster_std=1.0, n_features=2, random_state=None):
    """Random clusters: returns X and true labels y."""
    rng = np.random.default_rng(random_state)
    ctrs = rng.uniform(-10, 10, size=(centers, n_features))
    y = np.arange(n_samples) % centers
    X = ctrs[y] + rng.normal(0, cluster_std, size=(n_samples, n_features))
    return X, y


class GaussianMixture:
    """GMM with full covariance, trained by the EM algorithm."""
    def __init__(self, n_components=2, max_iter=100, tol=1e-3, reg_covar=1e-6, random_state=0):
        self.k = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg_covar
        self.random_state = random_state

    def _kmeans_init(self, X, rng):
        """k-means (k-means++ start) gives the initial responsibilities."""
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, self.k):
            d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
            p = d2 / d2.sum() if d2.sum() > 0 else None
            centers.append(X[rng.choice(len(X), p=p)])
        centers = np.array(centers, dtype=float)
        for _ in range(100):
            labels = ((X[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
            new = np.array([X[labels == j].mean(0) if np.any(labels == j) else centers[j]
                            for j in range(self.k)])
            if np.allclose(new, centers):
                break
            centers = new
        resp = np.zeros((len(X), self.k))
        resp[np.arange(len(X)), labels] = 1.0
        return resp

    def _m_step(self, X, resp):
        n, d = X.shape
        nk = resp.sum(0) + 1e-10
        self.weights_ = nk / n
        self.means_ = (resp.T @ X) / nk[:, None]
        self.covariances_ = np.zeros((self.k, d, d))
        for j in range(self.k):
            diff = X - self.means_[j]
            self.covariances_[j] = (resp[:, j, None] * diff).T @ diff / nk[j] + self.reg * np.eye(d)

    def _log_prob(self, X):
        """log( weight_k * N(x | mean_k, cov_k) ) for every sample and component."""
        n, d = X.shape
        out = np.zeros((n, self.k))
        for j in range(self.k):
            diff = X - self.means_[j]
            cov = self.covariances_[j]
            sign, logdet = np.linalg.slogdet(cov)
            maha = np.einsum('ij,jk,ik->i', diff, np.linalg.inv(cov), diff)
            out[:, j] = np.log(self.weights_[j]) - 0.5 * (d * np.log(2 * np.pi) + logdet + maha)
        return out

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        resp = self._kmeans_init(X, rng)
        self._m_step(X, resp)
        prev = -np.inf
        for _ in range(self.max_iter):
            lp = self._log_prob(X)                               # E-step
            m = lp.max(1, keepdims=True)
            lse = m + np.log(np.exp(lp - m).sum(1, keepdims=True))
            resp = np.exp(lp - lse)
            self._m_step(X, resp)                                # M-step
            ll = lse.mean()
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        return self

    def predict(self, X):
        return self._log_prob(np.asarray(X, dtype=float)).argmax(1)


# ==========================================================
# LISTING 1 : GMM on the small dataset (2 components)
# ==========================================================
print("LISTING 1: GMM ON SMALL DATASET")

data = [[2, 4], [4, 6], [6, 8], [10, 4], [12, 4]]
df = pd.DataFrame(data, columns=['X', 'Y'])

gmm = GaussianMixture(n_components=2, random_state=42)
gmm.fit(df.values)

labels = gmm.predict(df.values)
df['labels'] = labels
print(df)

print("\nMixing weights:", gmm.weights_.round(3))
print("Cluster means:\n", gmm.means_.round(3))


# ==========================================================
# LISTING 2 : GMM on a randomly generated dataset
# ==========================================================
print("\nLISTING 2: GMM ON RANDOM DATASET")

X, y = make_blobs(n_samples=200, centers=4, cluster_std=1, n_features=2, random_state=42)

df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
print("Data before clustering (first 5 rows):")
print(df.head())
print("\nTrue label counts:\n", df['label'].value_counts().sort_index())

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X)

labels = gmm.predict(X)
df['cluster'] = labels

print("\nData after clustering (first 5 rows):")
print(df.head())
print("\nCluster sizes:\n", df['cluster'].value_counts().sort_index())
print("\nMixing weights:", gmm.weights_.round(3))
print("Cluster means:\n", gmm.means_.round(3))

# Which true blob falls in which predicted cluster (replaces the coloured scatter)
print("\nTrue label vs predicted cluster:")
print(pd.crosstab(df['label'], df['cluster']))