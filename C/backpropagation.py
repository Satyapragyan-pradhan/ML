import numpy as np
import pandas as pd

# ================= HELPER FUNCTIONS (replace sklearn) =================
def split_indices(strat, test_size=0.25, seed=42):
    """Stratified split -> returns train indices and test indices."""
    rng = np.random.default_rng(seed)
    strat = np.asarray(strat)
    test_idx = []
    for c in np.unique(strat):
        idx = np.where(strat == c)[0]
        rng.shuffle(idx)
        test_idx.extend(idx[:int(round(len(idx) * test_size))])
    test_idx = np.array(sorted(test_idx))
    train_idx = np.setdiff1d(np.arange(len(strat)), test_idx)
    return train_idx, test_idx

def fit_scaler(X):
    mean, std = X.mean(axis=0), X.std(axis=0)      # ddof=0 like StandardScaler
    std[std == 0] = 1.0
    return mean, std


# ==========================================================
# TASK 2 : Split dataset into X (independent) and Y (dependent)
# ==========================================================
print("TASK 2: SPLIT X AND Y")

file_name = 'iris.csv'      # or: file_name = input("Enter dataset file name: ")
df = pd.read_csv(file_name)
print("Dataset (first 5 rows):")
print(df.head())
print("Shape of dataset:", df.shape)

X = df.iloc[:, :-1]     # all columns except the last
Y = df.iloc[:, -1]      # last column

print("\nIndependent variables X (first 5 rows):")
print(X.head())
print("Shape of X:", X.shape)
print("\nDependent variable Y (first 5 values):")
print(Y.head())
print("Shape of Y:", Y.shape)


# ==========================================================
# TASK 3 : DataFrame, statistics and preprocessing
# ==========================================================
print("\nTASK 3: DATAFRAME AND PREPROCESSING")

data = {
    'Name': ['Ravi', 'Anu', 'Kiran', 'Sita', 'Mohan', 'Lata', 'Raj', 'Meena', 'Arun', 'Divya',
             'Anu', 'Vijay'],
    'Age': [25, 30, 35, np.nan, 45, 28, 32, 40, 29, 33, 30, 38],
    'Salary': [30000, 45000, 52000, 61000, np.nan, 38000, 47000, 72000, 35000, 58000, 45000, 66000],
    'Experience': [2, 5, 8, 10, 15, 3, 6, 14, 4, 9, 5, 12],
    'Purchased': [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1]
}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# (a) Retrieve a particular column
print("\n(a) Age column:")
print(df['Age'])

# (b) Summary
print("\n(b) Summary statistics:")
print(df.describe())
print()
df.info()

# (c) Mean and standard deviation
print("\n(c) Mean of each numeric column:")
print(df.mean(numeric_only=True))
print("\n(c) Standard deviation of each numeric column:")
print(df.std(numeric_only=True))

# (d) Preprocessing
# 1. Remove duplicates
print("\n(d) Number of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
print("Shape after removing duplicates:", df.shape)

# 2. Missing values
print("\nMissing values per column:")
print(df.isnull().sum())
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
print("\nMissing values after filling with mean:")
print(df.isnull().sum())

features = ['Age', 'Salary', 'Experience']

# 3. Standardization (mean = 0, std = 1)
std_df = (df[features] - df[features].mean()) / df[features].std(ddof=0)
print("\nStandardized data (first 5 rows):")
print(std_df.head().round(3))
print("Mean after standardization:", std_df.mean().round(3).tolist())
print("Std after standardization:", std_df.std(ddof=0).round(3).tolist())

# 4. Normalization (0 - 1)
norm_df = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
print("\nNormalized data (first 5 rows):")
print(norm_df.head().round(3))
print("Min:", norm_df.min().tolist(), " Max:", norm_df.max().tolist())

# 5. Data balancing (random over-sampling of minority class)
print("\nClass distribution before balancing:")
print(df['Purchased'].value_counts())

majority = df[df['Purchased'] == 0]
minority = df[df['Purchased'] == 1]
minority_up = minority.sample(n=len(majority), replace=True, random_state=42)
df_bal = pd.concat([majority, minority_up]).reset_index(drop=True)

print("\nClass distribution after balancing:")
print(df_bal['Purchased'].value_counts())

# (e) Train / test split (stratified, 75% / 25%)
X = df_bal[features]
Y = df_bal['Purchased']
train_idx, test_idx = split_indices(Y.values, test_size=0.25, seed=42)
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

print("\n(e) Shapes -> X_train:", X_train.shape, " X_test:", X_test.shape,
      " Y_train:", Y_train.shape, " Y_test:", Y_test.shape)
print("\nX_train:")
print(X_train)
print("\nY_train:")
print(Y_train.values)


# ==========================================================
# TASK 4 : Perceptron
# ==========================================================
print("\nTASK 4: PERCEPTRON")

class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs

    def activation(self, z):            # step function
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0.0
        self.epochs_used = self.epochs
        for epoch in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = self.activation(np.dot(xi, self.w) + self.b)
                update = self.lr * (target - pred)      # perceptron learning rule
                self.w += update * xi
                self.b += update
                errors += int(update != 0)
            if errors == 0:                              # converged
                self.epochs_used = epoch + 1
                break
        return self

    def predict(self, X):
        return self.activation(np.dot(X, self.w) + self.b)

# Logic gates
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
gates = {
    "AND":  np.array([0, 0, 0, 1]),
    "OR":   np.array([0, 1, 1, 1]),
    "NAND": np.array([1, 1, 1, 0]),
    "NOR":  np.array([1, 0, 0, 0]),
    "XOR":  np.array([0, 1, 1, 0]),
}
for name, y in gates.items():
    p = Perceptron(lr=0.1, epochs=100).fit(X, y)
    pred = p.predict(X)
    print(f"{name:5s} weights={p.w.round(2)} bias={p.b:.2f} epochs={p.epochs_used:3d} "
          f"predictions={pred} accuracy={(pred == y).mean()*100:.0f}%")
print("Note: XOR is not linearly separable, so a single perceptron cannot learn it.")

# Iris: setosa vs versicolor (works whether or not the file has an Id column)
iris = pd.read_csv('iris.csv')
species = iris.iloc[:, -1].astype(str)
mask = species.str.contains('setosa') | species.str.contains('versicolor')
Xi = iris.iloc[:, -3:-1][mask].values.astype(float)          # petal length, petal width
yi = species[mask].str.contains('versicolor').astype(int).values

p = Perceptron(lr=0.01, epochs=100).fit(Xi, yi)
acc = (p.predict(Xi) == yi).mean() * 100
print(f"\nIris (setosa vs versicolor): weights={p.w.round(3)} bias={p.b:.3f} "
      f"epochs={p.epochs_used} accuracy={acc:.1f}%")


# ==========================================================
# TASK 5 : Neural Network with Backpropagation
# ==========================================================
print("\nTASK 5: NEURAL NETWORK (BACKPROPAGATION)")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class NeuralNetwork:
    """Input -> hidden (sigmoid) -> output (sigmoid), trained by backpropagation."""
    def __init__(self, n_in, n_hidden, n_out, lr=0.5, seed=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1, (n_in, n_hidden)); self.b1 = np.zeros((1, n_hidden))
        self.W2 = rng.normal(0, 1, (n_hidden, n_out)); self.b2 = np.zeros((1, n_out))
        self.lr = lr

    def forward(self, X):
        self.a1 = sigmoid(X @ self.W1 + self.b1)
        self.a2 = sigmoid(self.a1 @ self.W2 + self.b2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[0]
        d2 = (self.a2 - y) * self.a2 * (1 - self.a2)               # output error
        dW2 = self.a1.T @ d2 / m; db2 = d2.mean(axis=0, keepdims=True)
        d1 = (d2 @ self.W2.T) * self.a1 * (1 - self.a1)            # hidden error
        dW1 = X.T @ d1 / m; db1 = d1.mean(axis=0, keepdims=True)
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2         # gradient descent
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def train(self, X, y, epochs=5000):
        losses = []
        for _ in range(epochs):
            out = self.forward(X)
            losses.append(np.mean((y - out) ** 2))
            self.backward(X, y)
        return losses

# Test 1 : XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(2, 4, 1, lr=1.0)
loss_xor = nn.train(X, y, epochs=10000)
print("XOR problem")
print("Final loss:", round(loss_xor[-1], 5))
print("Loss at epochs 0, 1000, 5000:", [round(loss_xor[i], 5) for i in (0, 1000, 5000)])
print("Predicted probabilities:", nn.forward(X).ravel().round(3))
print("Predicted classes :", (nn.forward(X) > 0.5).astype(int).ravel())
print("Expected classes  :", y.ravel())

# Test 2 : Iris (3 classes)
iris = pd.read_csv('iris.csv')
Xi = iris.iloc[:, -5:-1].values.astype(float)       # last 4 columns before species
species = iris.iloc[:, -1].values
classes = pd.unique(species)
yi = np.array([[1 if s == c else 0 for c in classes] for s in species])   # one-hot

tr, te = split_indices(species, test_size=0.3, seed=42)
Xtr, Xte, ytr, yte = Xi[tr], Xi[te], yi[tr], yi[te]

mean, std = fit_scaler(Xtr)
Xtr, Xte = (Xtr - mean) / std, (Xte - mean) / std

nn2 = NeuralNetwork(4, 8, 3, lr=0.5, seed=3)
loss_iris = nn2.train(Xtr, ytr, epochs=3000)
train_acc = (nn2.forward(Xtr).argmax(1) == ytr.argmax(1)).mean() * 100
test_acc = (nn2.forward(Xte).argmax(1) == yte.argmax(1)).mean() * 100
print("\nIris dataset (4 inputs - 8 hidden - 3 outputs)")
print("Final training loss:", round(loss_iris[-1], 5))
print(f"Training accuracy: {train_acc:.2f}%")
print(f"Testing accuracy: {test_acc:.2f}%")


# ==========================================================
# TASK 6 : Fit hyperplane with Linear Regression, residual error
# ==========================================================
print("\nTASK 6: HYPERPLANE FIT AND RESIDUAL ERROR")

def fit_hyperplane(X, y):
    """y = b0 + b1*x1 + ... + bN*xN using the normal equation."""
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    return np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ y

def residual_errors(X, y, beta):
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    residuals = y - Xb @ beta
    return residuals, residuals.sum(), (residuals ** 2).sum()

# Example 1 : N = 2
X = np.array([[1, 2], [2, 1], [3, 4], [4, 3], [5, 6], [6, 5]], dtype=float)
y = np.array([5.1, 5.9, 10.8, 11.1, 16.2, 16.8])
beta = fit_hyperplane(X, y)
res, sum_res, sse = residual_errors(X, y, beta)
print("Example 1 (N = 2)")
print("Hyperplane coefficients [b0, b1, b2]:", beta.round(4))
print("Residuals:", res.round(4))
print("Sum of residuals :", round(sum_res, 6))
print("Sum of squared residuals :", round(sse, 6))

# Example 2 : N = 5, 100 points
rng = np.random.default_rng(0)
N = 5
X = rng.normal(size=(100, N))
true_beta = np.array([2.0, 1.5, -2.0, 0.5, 3.0, -1.0])
y = true_beta[0] + X @ true_beta[1:] + rng.normal(0, 0.5, 100)
beta = fit_hyperplane(X, y)
res, sum_res, sse = residual_errors(X, y, beta)
print("\nExample 2 (N = 5, 100 points)")
print("True coefficients  :", true_beta)
print("Fitted coefficients:", beta.round(4))
print("Sum of residuals :", round(sum_res, 6), "(~0 because the model has an intercept)")
print("Sum of squared residuals (SSE):", round(sse, 4))

# Verification with numpy least squares (replaces sklearn check)
Xb = np.hstack([np.ones((X.shape[0], 1)), X])
beta_np = np.linalg.lstsq(Xb, y, rcond=None)[0]
sse_np = ((y - Xb @ beta_np) ** 2).sum()
print("\nnumpy lstsq coefficients:", beta_np.round(4))
print("numpy SSE:", round(sse_np, 4), "| matches our SSE:", np.isclose(sse, sse_np))