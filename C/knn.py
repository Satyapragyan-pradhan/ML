import numpy as np
import pandas as pd

# ================= HELPER FUNCTIONS (replace sklearn) =================
def train_test_split(X, y, test_size=0.2, random_state=0):
    np.random.seed(random_state)
    idx = np.random.permutation(len(X))
    n_test = int(np.ceil(len(X) * test_size))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def fit_scaler(X_train):
    # same as StandardScaler: population std (ddof=0), fit on training data only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std

def transform(X, mean, std):
    return (X - mean) / std

def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        dist = np.sqrt(((X_train - x) ** 2).sum(axis=1))   # Euclidean distance
        nearest = np.argsort(dist)[:k]
        labels, counts = np.unique(y_train[nearest], return_counts=True)
        preds.append(labels[np.argmax(counts)])             # majority vote
    return np.array(preds)

def confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[np.where(classes == t)[0][0], np.where(classes == p)[0][0]] += 1
    return cm, classes

def classification_report(y_true, y_pred):
    cm, classes = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    precision = np.divide(tp, cm.sum(axis=0), out=np.zeros_like(tp), where=cm.sum(axis=0) != 0)
    recall = np.divide(tp, cm.sum(axis=1), out=np.zeros_like(tp), where=cm.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) != 0)
    support = cm.sum(axis=1)
    report = pd.DataFrame({'precision': precision, 'recall': recall,
                           'f1-score': f1, 'support': support}, index=classes)
    report.loc['macro avg'] = [precision.mean(), recall.mean(), f1.mean(), support.sum()]
    report.loc['weighted avg'] = [np.average(precision, weights=support),
                                  np.average(recall, weights=support),
                                  np.average(f1, weights=support), support.sum()]
    print(report.round(2))
    print(f"accuracy: {tp.sum() / cm.sum():.2f}")

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


# ==========================================================
# PART 1 : k-NN on Student Result Dataset
# ==========================================================
print("PART 1: k-NN ON STUDENT RESULT DATASET")

# Save this as "knearest csv.csv" (or create the DataFrame directly below)
dataset = pd.read_csv("knearest csv.csv")
# If no file is available, use this instead:
# dataset = pd.DataFrame({
#     'CGPA': [9.2, 8, 8.5, 6, 6.5, 8.2, 5.8, 8.9],
#     'Assessment': [85, 80, 81, 45, 50, 72, 38, 91],
#     'Project': [8, 7, 8, 5, 4, 7, 5, 9],
#     'Result': ['Pass', 'Pass', 'Pass', 'Fail', 'Fail', 'Pass', 'Fail', 'Pass']})

print(dataset.shape)
print(dataset.head())
print(dataset.tail(2))

X = dataset.iloc[:, :-1].values.astype(float)
y = dataset.iloc[:, 3].to_numpy()
print(X)
print(y)

# Split 80% / 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Standardise (fit on training data only)
mean, std = fit_scaler(X_train)
X_train_s = transform(X_train, mean, std)
X_test_s = transform(X_test, mean, std)

# k-NN with k = 3
y_pred = knn_predict(X_train_s, y_train, X_test_s, k=3)

print("\nCorrect and wrong predictions:")
for actual, predicted in zip(y_test, y_pred):
    status = "CORRECT" if actual == predicted else "WRONG"
    print("Actual:", actual, "| Predicted:", predicted, "->", status)

cm, classes = confusion_matrix(y_test, y_pred)
print(cm)
classification_report(y_test, y_pred)

# Classify the new sample (scaled the same way as training data)
new_sample = np.array([[9.1, 85, 8.]])
print([9.1, 85, 8.])
print(knn_predict(X_train_s, y_train, transform(new_sample, mean, std), k=3))


# ==========================================================
# PART 2 : k-NN on Iris Dataset
# ==========================================================
print("\nPART 2: k-NN ON IRIS DATASET")

dataset = pd.read_csv("Iris.csv")
print(dataset.shape)
print(dataset.head())
print(dataset.tail(2))
print(dataset.describe())

X = dataset.iloc[:, 1:5].values.astype(float)   # skip Id column
y = dataset.iloc[:, 5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

y_pred = knn_predict(X_train, y_train, X_test, k=3)

cm, classes = confusion_matrix(y_test, y_pred)
print(cm)
classification_report(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred) * 100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')