import numpy as np
import pandas as pd

# ================= HELPER FUNCTIONS (replace sklearn) =================
class KMeans:
    def __init__(self, n_clusters=3, n_init=10, max_iter=300, random_state=0):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state

    def _init_centers(self, X, rng):
        """k-means++ initialisation."""
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, self.n_clusters):
            d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centers], axis=0)
            if d2.sum() == 0:
                centers.append(X[rng.integers(len(X))])
            else:
                centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        return np.array(centers, dtype=float)

    def _single_run(self, X, rng):
        centers = self._init_centers(X, rng)
        for _ in range(self.max_iter):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = d.argmin(axis=1)
            new_centers = centers.copy()
            for k in range(self.n_clusters):
                if np.any(labels == k):
                    new_centers[k] = X[labels == k].mean(axis=0)
                else:   # empty cluster -> move to the farthest point
                    new_centers[k] = X[d.min(axis=1).argmax()]
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = d.argmin(axis=1)
        inertia = d.min(axis=1).sum()
        return centers, labels, inertia

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):               # keep the best of n_init runs
            run = self._single_run(X, rng)
            if best is None or run[2] < best[2]:
                best = run
        self.cluster_centers_, self.labels_, self.inertia_ = best
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        d = ((X[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return d.argmin(axis=1)


def silhouette_score(X, labels):
    """Mean silhouette coefficient (Euclidean distance)."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    uniq = np.unique(labels)
    s = np.zeros(n)
    for i in range(n):
        same = labels == labels[i]
        if same.sum() <= 1:
            s[i] = 0.0                              # singleton cluster
            continue
        a = D[i, same].sum() / (same.sum() - 1)     # mean intra-cluster distance
        b = min(D[i, labels == c].mean() for c in uniq if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return s.mean()


# ==========================================================
# LISTING 1 : k-Means on the small dataset (k = 2)
# ==========================================================
print("LISTING 1: K-MEANS ON SMALL DATASET")

data = [[2, 4], [4, 6], [6, 8], [10, 4], [12, 4]]
df = pd.DataFrame(data, columns=['X', 'Y'])
print(df)

kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
kmeans.fit(df.values)
df['pred'] = kmeans.predict(df[['X', 'Y']].values)

print("\nCluster assignment:")
print(df)
print("\nCluster centres:")
print(kmeans.cluster_centers_)

labels = kmeans.labels_

# As in the textbook: df here also contains the 'pred' column
score = silhouette_score(df.values, labels)
print('\nSilhouette score is:', score)

# Silhouette score using only the X and Y features
score_xy = silhouette_score(df[['X', 'Y']].values, labels)
print('Silhouette score using only X, Y is:', score_xy)


# ==========================================================
# LISTING 2 : k-Means on Iris (k = 3, first two features)
# ==========================================================
print("\nLISTING 2: K-MEANS ON IRIS")

iris = pd.read_csv('iris.csv')
# drop an Id column if the file has one
iris = iris.drop(columns=[c for c in iris.columns if c.lower() == 'id'])

df = iris.copy()
df = df.iloc[:, [0, 1, -1]]            # first two features and the species column
df.columns = ['x1', 'x2', 'y']
print(df.head(10))

kmeans = KMeans(n_clusters=3, n_init=10, random_state=0)
X = df[['x1', 'x2']].values
kmeans.fit(X)
df['pred'] = kmeans.predict(X)

print("\nCluster sizes:", np.bincount(df['pred']))
print("\nCluster centres:")
print(kmeans.cluster_centers_.round(3))

score = silhouette_score(X, kmeans.labels_)
print('\nSilhouette score is:', score)