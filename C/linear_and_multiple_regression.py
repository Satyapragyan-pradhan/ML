import numpy as np
import pandas as pd

# ================= HELPER FUNCTIONS (replace sklearn) =================
def fit_linear_regression(X, y):
    """Least squares fit. X is 2-D (n_samples x n_features). Returns intercept, coefficients."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_b = np.column_stack([np.ones(len(X)), X])        # add column of 1s for intercept
    theta = np.linalg.lstsq(X_b, y, rcond=None)[0]     # same as (X'X)^-1 X'y
    return theta[0], theta[1:]

def predict(X, intercept, coef):
    return np.asarray(X, dtype=float) @ coef + intercept

def r2_score(y, y_pred):
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - ss_res / ss_tot

def make_regression(n_samples=50, n_features=1, noise=0.1):
    """Random linear dataset (replaces sklearn.datasets.make_regression)."""
    X = np.random.randn(n_samples, n_features)
    true_coef = 100 * np.random.rand(n_features)
    y = X @ true_coef + noise * np.random.randn(n_samples)
    return X, y


# ==========================================================
# PART 1 : Linear Regression on Sales Data
# ==========================================================
print("PART 1: LINEAR REGRESSION ON SALES DATA")

salesdata = {'week': [1, 2, 3, 4, 5],
             'sales': [1.2, 1.8, 2.6, 3.2, 3.8]}
df = pd.DataFrame(salesdata, columns=['week', 'sales'])

X = df[['week']].values        # independent variable (2-D)
y = df['sales'].values         # dependent variable

intercept, coef = fit_linear_regression(X, y)

print('Intercept: \n', intercept)
print('Coefficients: \n', coef)
print('\nThe Regression Equation is', coef, '* X +', intercept)

pred = predict(X, intercept, coef)
print("\nR Squared for Regression model:", r2_score(y, pred))


# ==========================================================
# PART 2 : Linear Regression on Random Dataset
# ==========================================================
print("\nPART 2: LINEAR REGRESSION ON RANDOM DATASET")

X, y = make_regression(n_samples=50, n_features=1, noise=0.1)

intercept, coef = fit_linear_regression(X, y)

print('Intercept: \n', intercept)
print('Coefficients: \n', coef)
print('\nThe Regression Equation is', coef, '* X +', intercept)

pred = predict(X, intercept, coef)
print("\nR Squared for Regression model:", r2_score(y, pred))


# ==========================================================
# PART 3 : Multiple Regression
# ==========================================================
print("\nPART 3: MULTIPLE REGRESSION")

X, y = make_regression(n_samples=50, n_features=2, noise=0.1)

intercept, coef = fit_linear_regression(X, y)

print('Intercept: \n', intercept)
print('Coefficients: \n', coef)

pred = predict(X, intercept, coef)
print("\nR Squared for Regression model:", r2_score(y, pred))