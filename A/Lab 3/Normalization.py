from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data.tolist()

# Apply Min-Max normalization
def normalize(X):
    rows = len(X)
    columns = len(X[0])

    minimum = []
    maximum = []
    
    # Find minimum and maximum of each feature
    for j in range(columns):
        low = X[0][j]
        high = X[0][j]
        for i in range(rows):
            if X[i][j] < low:
                low = X[i][j]

            if X[i][j] > high:
                high = X[i][j]

        minimum.append(low)
        maximum.append(high)

    # Scale values between 0 and 1
    normalized = []
    for i in range(rows):
        new_row = []
        for j in range(columns):
            difference = maximum[j] - minimum[j]
            if difference != 0:
                value = ((X[i][j] - minimum[j]) / difference)
            else:
                value = 0.0
            new_row.append(value)
        normalized.append(new_row)
        
    return normalized

# Main program
X_normalized = normalize(X)

print("Original First Sample:", X[0])
print("Normalized First Sample:", X_normalized[0])
print("Total Samples:", len(X_normalized))
print("Total Features:", len(X_normalized[0]))