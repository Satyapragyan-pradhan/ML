from sklearn.datasets import load_iris
import math

# Load Iris dataset
iris = load_iris()
X = iris.data.tolist()

# Split data manually into training and testing sets
def split_data(X):
    X_train, X_test = [], []
    train_count = int(len(X) * 0.70)
    for i in range(len(X)):
        if i < train_count:
            X_train.append(X[i])
        else:
            X_test.append(X[i])

    return X_train, X_test

# Calculate mean
def calculate_mean(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)

# Train simple linear regression
def train_regression(X_train):
    x_values = []
    y_values = []
    for row in X_train:
        x_values.append(row[1])  # Sepal width
        y_values.append(row[0])  # Sepal length

    mean_x = calculate_mean(x_values)
    mean_y = calculate_mean(y_values)

    numerator = 0.0
    denominator = 0.0
    for i in range(len(x_values)):
        numerator += ((x_values[i] - mean_x) * (y_values[i] - mean_y))
        denominator += ((x_values[i] - mean_x) ** 2)

    slope = numerator / denominator
    intercept = mean_y - slope * mean_x

    return slope, intercept

# Predict target value
def predict(x, slope, intercept):
    return slope * x + intercept

# Calculate Mean Squared Error
def calculate_mse(actual, predicted):
    total = 0.0
    for i in range(len(actual)):
        difference = actual[i] - predicted[i]
        total += difference * difference

    return total / len(actual)

# Calculate Root Mean Squared Error
def calculate_rmse(mse):
    return math.sqrt(mse)

# Calculate R-squared
def calculate_r2(actual, predicted):
    mean_y = calculate_mean(actual)
    total_error = 0.0
    total_variation = 0.0

    for i in range(len(actual)):
        total_error += (actual[i] - predicted[i]) ** 2
        total_variation += (actual[i] - mean_y) ** 2

    if total_variation == 0:
        return 0.0

    return 1 - total_error / total_variation

# Main program
X_train, X_test = split_data(X)
slope, intercept = train_regression(X_train)

actual = []
predicted = []
for row in X_test:
    actual.append(row[0])
    predicted.append(predict(row[1], slope, intercept))

mse = calculate_mse(actual, predicted)
rmse = calculate_rmse(mse)
r2 = calculate_r2(actual, predicted)

print("Linear Regression")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Slope:", slope)
print("Intercept:", intercept)
print("MSE:", mse)
print("RMSE:", rmse)
print("R-squared:", r2)