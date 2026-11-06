from sklearn.datasets import load_iris
import math

# Load Iris dataset
iris = load_iris()
X = iris.data.tolist()
feature_names = iris.feature_names

# Calculate mean
def calculate_mean(values):
    total = 0.0
    for value in values:
        total += value
    return total / len(values)

# Calculate Pearson correlation coefficient
def pearson_correlation(X, Y):
    mean_x = calculate_mean(X)
    mean_y = calculate_mean(Y)

    numerator = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for i in range(len(X)):
        dx = X[i] - mean_x
        dy = Y[i] - mean_y

        numerator += dx * dy
        sum_x += dx * dx
        sum_y += dy * dy
    denominator = math.sqrt(sum_x * sum_y)

    if denominator == 0:
        return 0.0

    return numerator / denominator

# Calculate correlation matrix manually
def correlation_matrix(data):
    features = len(data[0])
    matrix = []
    for i in range(features):
        row = []
        for j in range(features):
            x_values = []
            y_values = []
            for sample in data:
                x_values.append(sample[i])
                y_values.append(sample[j])

            correlation = pearson_correlation(x_values, y_values)
            row.append(correlation)
        matrix.append(row)

    return matrix

# Main program
matrix = correlation_matrix(X)
print("Pearson Correlation Matrix")

for row in matrix:
    print("[", end=" ")
    for value in row:
        print(round(value, 3), end=" ")
    print("]")

print("\nFeature Names:")
for name in feature_names:
    print(name)