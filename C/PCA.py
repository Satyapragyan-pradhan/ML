import numpy as np
import pandas as pd
from numpy.linalg import eig

# ================= HELPER FUNCTIONS (replace sklearn) =================
class PCA:
    """PCA using SVD (same method sklearn uses)."""
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        self.mean_ = X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        # sign fix so results are repeatable
        idx = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[idx, np.arange(U.shape[1])])
        signs[signs == 0] = 1
        Vt = Vt * signs[:, None]
        self.components_ = Vt[:self.n_components]
        ev = (S ** 2) / (n - 1)
        self.explained_variance_ = ev[:self.n_components]
        self.explained_variance_ratio_ = (ev / ev.sum())[:self.n_components]
        self.singular_values_ = S[:self.n_components]
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_


def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dist = np.sqrt(((X_train - x) ** 2).sum(axis=1))
        nearest = np.argsort(dist, kind='stable')[:k]
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        preds.append(labels[np.argmax(counts)])
    return np.array(preds)


def cross_val_score_knn(X, y, k=3, cv=10):
    """Stratified k-fold cross validation (no shuffle), returns accuracy per fold."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    fold_of = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        for f, chunk in enumerate(np.array_split(idx, cv)):
            fold_of[chunk] = f
    scores = []
    for f in range(cv):
        test = fold_of == f
        train = ~test
        pred = knn_predict(X[train], y[train], X[test], k)
        scores.append(np.mean(pred == y[test]))
    return np.array(scores)


# ==========================================================
# TASK 1 : PCA and Inverse Transform
# ==========================================================
print("TASK 1: PCA AND INVERSE TRANSFORM")

X = np.array([[2, 6], [1, 7]])
print("Original Matrix X and its Shape")
print(X)
print("Original Shape:", X.shape)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("\nTransformed Matrix and its Shape")
print(X_pca)
print("Transformed Shape:", X_pca.shape)

X_new = pca.inverse_transform(X_pca)
print("\nAfter Inverse Transform")
print(X_new)

print("\nExplained variance ratio")
print(pca.explained_variance_ratio_)
print("\nSingular values")
print(pca.singular_values_)

print("\nAre X and inverse-transformed X similar? ->", np.allclose(X, X_new))

# Same experiment with the matrix exactly as printed in the textbook
X2 = np.array([[2, 1], [6, 7]])
pca2 = PCA(n_components=2)
X2_new = pca2.inverse_transform(pca2.fit_transform(X2))
print("\nTextbook matrix [[2,1],[6,7]] reconstructed:")
print(X2_new)
print("Similar? ->", np.allclose(X2, X2_new))


# ==========================================================
# TASK 2 : PCA step by step
# ==========================================================
print("\nTASK 2: PCA STEP BY STEP")

X = np.array([[3, 6], [4, 7]])
print("Original Matrix X and its Shape")
print(X)
print("Original Shape:", X.shape)

# Mean of each column
M = np.mean(X.T, axis=1)
print("\nMean matrix")
print(M)

# Centre the columns
C = X - M
print("\nCentre the matrix")
print(C)

# Covariance matrix of the centred data
V = np.cov(C.T)
print("\nCovariance of the matrix")
print(V)

# Eigen decomposition
values, vectors = eig(V)
print("\nEigen vectors")
print(vectors)
print("\nEigen values")
print(values)

# Project data onto eigenvectors
P = vectors.T.dot(C.T)
print("\nProjected data (principal component scores)")
print(P.T)


# ==========================================================
# TASK 3 : PCA on Iris, KNN (k=3) before and after PCA
# ==========================================================
print("\nTASK 3: PCA ON IRIS DATASET")

df = pd.read_csv("iris.csv")
print(df.head(10))

array = df.values
X = array[:, 0:4].astype(float)    # if file has an Id column use array[:, 1:5], labels array[:, 5]
y = array[:, 4]

score = cross_val_score_knn(X, y, k=3, cv=10)
print("\nCross score before applying PCA\n")
print(score.mean())

print("\nApply PCA now...")
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)
score_pca = cross_val_score_knn(X_pca, y, k=3, cv=10)
print("\nCross score after applying PCA (1 component)\n")
print(score_pca.mean())
print("Variance retained by 1 component:", pca.explained_variance_ratio_.sum())

print("\nComponents vs cross-validation score")
for k in range(1, 5):
    p = PCA(n_components=k)
    s = cross_val_score_knn(p.fit_transform(X), y, k=3, cv=10).mean()
    print(f"n_components={k}: score={s:.4f}, variance retained={p.explained_variance_ratio_.sum():.4f}")