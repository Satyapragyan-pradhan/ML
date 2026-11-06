import numpy as np

# ==========================================================
# PART A : Perceptron for Boolean AND
# ==========================================================
print("PART A: PERCEPTRON FOR AND FUNCTION")

class Perceptron(object):
    def __init__(self, input_size, lr=0.2, epochs=4):
        self.W = np.array([0.3, -0.2])
        self.epochs = epochs
        self.lr = lr

    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x, theta):
        z = self.W.T.dot(x) - theta
        z = round(z, 2)
        a = self.activation_fn(z)
        return a

    def fit(self, X, d, theta, count):
        for _ in range(self.epochs):
            print("Epoch: ", count)
            count = count + 1
            for i in range(d.shape[0]):
                x = X[i]
                print("input", x, "\t", "Weight:", self.W)
                y = self.predict(x, theta)
                e = d[i] - y
                self.W = self.W + self.lr * e * x
            print()

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
d = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
theta = 0.4
count = 1
perceptron.fit(X, d, theta, count)
print("Learned weights:", perceptron.W)

# Verify against the AND truth table
print("\nX1 X2 | Ydes | Ypred")
print("------------------------")
for xi, di in zip(X, d):
    print(f" {xi[0]}  {xi[1]}  |  {di}   |  {perceptron.predict(xi, theta)}")


# ==========================================================
# PART B : Perceptron outputs for w1 = 1.0, w2 = 1.0, theta = 1.5
# ==========================================================
print("\nPART B: PERCEPTRON OUTPUT TABLE")

w1, w2, theta = 1.0, 1.0, 1.5

def step(z):
    return 1 if z >= 0 else 0

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("x1 x2 | z = w1*x1 + w2*x2 - theta | output y")
print("-" * 52)
outputs = []
for x1, x2 in inputs:
    z = w1 * x1 + w2 * x2 - theta
    y = step(z)
    outputs.append(y)
    print(f"{x1}  {x2}  | {z:>14.1f} | {y}")

print("\nDecision boundary: x1 + x2 = 1.5")
print("Output is 1 only for (1,1) -> Boolean AND function")


# ==========================================================
# PART C : Gender classification using height and weight
# ==========================================================
print("\nPART C: GENDER CLASSIFICATION")

# Step 1: Create the dataset
np.random.seed(42)
n = 100   # samples per class

# Females (label 0)
h_f = np.random.normal(162, 6, n)
w_f = np.random.normal(58, 7, n)
# Males (label 1)
h_m = np.random.normal(176, 6, n)
w_m = np.random.normal(76, 8, n)

X = np.vstack([np.column_stack([h_f, w_f]), np.column_stack([h_m, w_m])])
y = np.array([0] * n + [1] * n)

# shuffle and split 80% train / 20% test
idx = np.random.permutation(len(y))
X, y = X[idx], y[idx]
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("Training samples:", X_train.shape[0], "| Test samples:", X_test.shape[0])
print("First 5 samples (height cm, weight kg, label 0=Female 1=Male):")
for xi, yi in zip(X_train[:5], y_train[:5]):
    print(f"  {xi[0]:6.1f} {xi[1]:5.1f} -> {yi}")

# Step 2: Standardise the features
mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train_s = (X_train - mu) / sigma
X_test_s = (X_test - mu) / sigma
print("Mean (height, weight):", mu.round(2))
print("Std (height, weight):", sigma.round(2))

# Step 3: Perceptron class
class GenderPerceptron:
    def __init__(self, n_features, lr=0.1, epochs=30):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs
        self.errors_per_epoch = []

    def activation(self, z):
        return 1 if z >= 0 else 0

    def predict_one(self, x):
        return self.activation(np.dot(self.w, x) + self.b)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def fit(self, X, d):
        for epoch in range(self.epochs):
            errors = 0
            for xi, di in zip(X, d):
                e = di - self.predict_one(xi)
                if e != 0:
                    self.w += self.lr * e * xi
                    self.b += self.lr * e
                    errors += 1
            self.errors_per_epoch.append(errors)
            if errors == 0:     # all training samples classified correctly
                print(f"Converged at epoch {epoch + 1}")
                break
        return self

# Step 4: Train the model
model = GenderPerceptron(n_features=2, lr=0.1, epochs=30)
model.fit(X_train_s, y_train)
print("Misclassifications per epoch:", model.errors_per_epoch)
print("Learned weights (standardised space):", model.w.round(4))
print("Learned bias:", round(model.b, 4))

# Step 5: Evaluate on training and test data
train_pred = model.predict(X_train_s)
test_pred = model.predict(X_test_s)
train_acc = (train_pred == y_train).mean() * 100
test_acc = (test_pred == y_test).mean() * 100
print(f"Training accuracy: {train_acc:.2f}%")
print(f"Test accuracy    : {test_acc:.2f}%")

tp = np.sum((test_pred == 1) & (y_test == 1))
tn = np.sum((test_pred == 0) & (y_test == 0))
fp = np.sum((test_pred == 1) & (y_test == 0))
fn = np.sum((test_pred == 0) & (y_test == 1))

print("\nConfusion matrix (test set)")
print("               Pred Female  Pred Male")
print(f"Actual Female   {tn:5d}      {fp:5d}")
print(f"Actual Male     {fn:5d}      {tp:5d}")

# Step 7: Predict the gender of new people
new_people = [(185, 85), (155, 50), (170, 65), (178, 72), (160, 60)]

print("\nHeight(cm)  Weight(kg)  -> Prediction")
print("-" * 40)
for h, wt in new_people:
    x_s = (np.array([h, wt]) - mu) / sigma
    label = "Male" if model.predict_one(x_s) == 1 else "Female"
    print(f"{h:>8}  {wt:>8}   -> {label}")