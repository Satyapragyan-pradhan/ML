import numpy as np
import pandas as pd

# ================= HELPER FUNCTIONS (replace scipy / sklearn) =================
def linkage(X, method='single'):
    """Agglomerative clustering. Returns Z like scipy: [cluster1, cluster2, distance, size]."""
    X = np.asarray(X, dtype=float)
    n = len(X)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))   # Euclidean distances
    members = {i: [i] for i in range(n)}

    def cdist(a, b):
        A, B = members[a], members[b]
        if method == 'single':
            return D[np.ix_(A, B)].min()
        if method == 'complete':
            return D[np.ix_(A, B)].max()
        if method == 'average':
            return D[np.ix_(A, B)].mean()
        if method == 'ward':
            ca, cb = X[A].mean(axis=0), X[B].mean(axis=0)
            na, nb = len(A), len(B)
            return np.sqrt(2 * na * nb / (na + nb)) * np.linalg.norm(ca - cb)
        raise ValueError("method must be single, complete, average or ward")

    cache = {(i, j): cdist(i, j) for i in range(n) for j in range(i + 1, n)}
    Z = []
    for step in range(n - 1):
        (a, b), d = min(cache.items(), key=lambda kv: (kv[1], kv[0]))   # closest pair
        new_id = n + step
        members[new_id] = members.pop(a) + members.pop(b)
        cache = {k: v for k, v in cache.items() if a not in k and b not in k}
        for c in members:
            if c != new_id:
                cache[(c, new_id)] = cdist(c, new_id)
        Z.append([a, b, d, len(members[new_id])])
    return np.array(Z)

def cut_tree(Z, n, n_clusters):
    """Cut the hierarchy into n_clusters; labels numbered by first appearance."""
    members = {i: [i] for i in range(n)}
    for step in range(n - n_clusters):
        a, b = int(Z[step, 0]), int(Z[step, 1])
        members[n + step] = members.pop(a) + members.pop(b)
    labels = np.zeros(n, dtype=int)
    for cid, (_, pts) in enumerate(sorted(members.items(), key=lambda kv: min(kv[1]))):
        labels[pts] = cid
    return labels

def agglomerative(X, n_clusters, method):
    X = np.asarray(X, dtype=float)
    return cut_tree(linkage(X, method), len(X), n_clusters)


# ==========================================================
# LISTING 1 : Linkage (merge tables replace the dendrograms)
# ==========================================================
print("LISTING 1: HIERARCHICAL CLUSTERING ON X")
X = np.array([[1, 4], [2, 8], [5, 10], [12, 18], [14, 28]])

for method in ['single', 'complete', 'average', 'ward']:
    Z = linkage(X, method)
    print(f"\n{method.capitalize()} Linkage Method")
    print(pd.DataFrame(Z, columns=['Cluster 1', 'Cluster 2', 'Distance', 'Size'])
          .astype({'Cluster 1': int, 'Cluster 2': int, 'Size': int}).round(4).to_string(index=False))


# ==========================================================
# LISTING 2 : Predict groups (single linkage, 2 clusters)
# ==========================================================
print("\nLISTING 2: AGGLOMERATIVE CLUSTERING (SINGLE, 2 CLUSTERS)")
data = np.array([[8, 12], [12, 17], [20, 20]])

print("The Predicted groupings for test data")
print("Test data are (8,12),(12,17),(20,20)")
pred = agglomerative(data, n_clusters=2, method='single')
print("\nCluster ID's are")
print(pred)


# ==========================================================
# LISTING 3 : Iris petal data, complete linkage, 3 clusters
# ==========================================================
print("\nLISTING 3: IRIS - COMPLETE LINKAGE")
iris = pd.read_csv('iris.csv')

# keep only petal length and petal width (the two columns before species)
iris = iris.iloc[:, -3:-1]
iris.columns = ['petal.length', 'petal.width']

labels = agglomerative(iris.values, n_clusters=3, method='complete')
print("Number of samples in each cluster:", np.bincount(labels))
print("\nMean petal length/width of each cluster:")
print(iris.groupby(labels).mean().round(3))


# ==========================================================
# LISTING 4 : All four linkages on Iris
# ==========================================================
print("\nLISTING 4: ALL FOUR LINKAGES ON IRIS")
for method in ['single', 'complete', 'average', 'ward']:
    labels = agglomerative(iris.values, n_clusters=3, method=method)
    print(method, '-> cluster sizes:', np.bincount(labels))