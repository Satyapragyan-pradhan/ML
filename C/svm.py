import numpy as np
import pandas as pd

# ================= HELPER FUNCTIONS (replace sklearn) =================
def train_test_split(X, y, test_size=0.2, random_state=0):
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))
    n_test = int(np.ceil(len(X) * test_size))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[np.where(classes == t)[0][0], np.where(classes == p)[0][0]] += 1
    return cm, classes

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def classification_report(y_true, y_pred):
    cm, classes = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    pc, rc = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pc, out=np.zeros_like(tp), where=pc != 0)
    recall = np.divide(tp, rc, out=np.zeros_like(tp), where=rc != 0)
    s = precision + recall
    f1 = np.divide(2 * precision * recall, s, out=np.zeros_like(tp), where=s != 0)
    rep = pd.DataFrame({'precision': precision, 'recall': recall,
                        'f1-score': f1, 'support': rc}, index=classes)
    rep.loc['macro avg'] = [precision.mean(), recall.mean(), f1.mean(), rc.sum()]
    rep.loc['weighted avg'] = [np.average(precision, weights=rc),
                               np.average(recall, weights=rc),
                               np.average(f1, weights=rc), rc.sum()]
    print(rep.round(2))
    print(f"accuracy: {tp.sum() / cm.sum():.2f}")


# ================= SVM FROM SCRATCH (replaces sklearn.svm.SVC) =================
class SVC:
    def __init__(self, kernel='linear', C=1.0, random_state=0):
        self.kernel = kernel
        self.C = C
        self.random_state = random_state

    def _kernel(self, A, B):
        if self.kernel == 'linear':
            return A @ B.T
        # rbf, gamma='scale' (same default as sklearn)
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-self.gamma * d)

    def _smo(self, K, y, tol=1e-3, max_passes=5, max_iter=300):
        """Simplified SMO for a binary SVM, labels y in {-1, +1}."""
        rng = np.random.default_rng(self.random_state)
        n = len(y)
        a = np.zeros(n)
        b = 0.0
        passes = 0
        it = 0
        while passes < max_passes and it < max_iter:
            it += 1
            changed = 0
            for i in range(n):
                Ei = (a * y) @ K[:, i] + b - y[i]
                if (y[i] * Ei < -tol and a[i] < self.C) or (y[i] * Ei > tol and a[i] > 0):
                    j = rng.integers(n - 1)
                    j = j + 1 if j >= i else j
                    Ej = (a * y) @ K[:, j] + b - y[j]
                    ai_old, aj_old = a[i], a[j]
                    if y[i] != y[j]:
                        L, H = max(0, aj_old - ai_old), min(self.C, self.C + aj_old - ai_old)
                    else:
                        L, H = max(0, ai_old + aj_old - self.C), min(self.C, ai_old + aj_old)
                    if L == H:
                        continue
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    a[j] = np.clip(aj_old - y[j] * (Ei - Ej) / eta, L, H)
                    if abs(a[j] - aj_old) < 1e-5:
                        continue
                    a[i] = ai_old + y[i] * y[j] * (aj_old - a[j])
                    b1 = b - Ei - y[i] * (a[i] - ai_old) * K[i, i] - y[j] * (a[j] - aj_old) * K[i, j]
                    b2 = b - Ej - y[i] * (a[i] - ai_old) * K[i, j] - y[j] * (a[j] - aj_old) * K[j, j]
                    if 0 < a[i] < self.C:
                        b = b1
                    elif 0 < a[j] < self.C:
                        b = b2
                    else:
                        b = (b1 + b2) / 2
                    changed += 1
            passes = passes + 1 if changed == 0 else 0
        return a, b

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.gamma = 1.0 / (X.shape[1] * X.var())
        self.classes_ = np.unique(y)
        self.models = []     # one binary model per pair of classes
        for p in range(len(self.classes_)):
            for q in range(p + 1, len(self.classes_)):
                m = (y == self.classes_[p]) | (y == self.classes_[q])
                Xs = X[m]
                ys = np.where(y[m] == self.classes_[p], 1.0, -1.0)
                a, b = self._smo(self._kernel(Xs, Xs), ys)
                sv = a > 1e-8
                self.models.append((p, q, Xs[sv], a[sv] * ys[sv], b))
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        votes = np.zeros((len(X), len(self.classes_)))
        for p, q, Xsv, ay, b in self.models:
            d = self._kernel(X, Xsv) @ ay + b
            votes[d >= 0, p] += 1
            votes[d < 0, q] += 1
        return self.classes_[np.argmax(votes, axis=1)]


# ==========================================================
# Step 2 : Read the Iris dataset
# ==========================================================
df = pd.read_csv("iris1.csv")      # 4 feature columns followed by species
print(df.head(10))
print("\nShape:", df.shape)
print("\nClass counts:\n", df.iloc[:, 4].value_counts())

# ==========================================================
# Step 3 : Features, labels and split (95% test)
# ==========================================================
array = df.values
X = array[:, 0:4].astype(float)
y = array[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=0)
print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# ==========================================================
# Step 4 : SVM with Linear kernel
# ==========================================================
model = SVC(kernel='linear', random_state=0)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n\nThe confusion matrix is \n")
conf, labels = confusion_matrix(y_test, pred)
print(conf)
print("\n\nAccuracy is")
print(accuracy_score(y_test, pred))
print("\n Classification Report")
classification_report(y_test, pred)

# ==========================================================
# Step 5 : SVM with Gaussian RBF kernel
# ==========================================================
model1 = SVC(kernel='rbf', random_state=0)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

print("\n\nThe confusion matrix for RBF kernel is \n")
conf1, _ = confusion_matrix(y_test, pred1)
print(conf1)
print("\n\nAccuracy for RBF kernel is")
print(accuracy_score(y_test, pred1))
print("\n Classification Report for RBF kernel")
classification_report(y_test, pred1)

# ==========================================================
# Additional : Effect of training-set size (table only)
# ==========================================================
print("\nTest size | Train samples | Linear acc. | RBF acc.")
print("-" * 52)
for ts in [0.95, 0.8, 0.5, 0.3, 0.2]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=0)
    a_lin = accuracy_score(yte, SVC(kernel='linear', random_state=0).fit(Xtr, ytr).predict(Xte))
    a_rbf = accuracy_score(yte, SVC(kernel='rbf', random_state=0).fit(Xtr, ytr).predict(Xte))
    print(f"  {ts:>4.2f}   |   {len(ytr):>4}        |   {a_lin:.4f}    |  {a_rbf:.4f}")