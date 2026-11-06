from sklearn.datasets import load_iris

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

# Calculate covariance between two features
def covariance(X, Y):
    mean_x = calculate_mean(X)
    mean_y = calculate_mean(Y)
    total = 0.0
    for i in range(len(X)):
        total += (X[i] - mean_x) * (Y[i] - mean_y)
    return total / (len(X) - 1)

# Calculate covariance matrix
def covariance_matrix(data):
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
            value = covariance(x_values, y_values)
            row.append(value)
        matrix.append(row)

    return matrix

# Main program
matrix = covariance_matrix(X)
print("Covariance Matrix:\n")

for row in matrix:
    for value in row:
        print(round(value, 3), end="  ")
    print()

print("\nFeature Names:")
for name in feature_names:
    print(name)