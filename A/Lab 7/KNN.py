from sklearn.datasets import load_iris
import math

# Load Iris dataset
iris = load_iris()
X = iris.data.tolist()
y = iris.target.tolist()

# Split data manually into training and testing sets
def split_data(X, y):
    X_train, X_test = [], []
    y_train, y_test = [], []

    for c in [0, 1, 2]:
        count = 0
        for label in y:
            if label == c:
                count += 1

        train_count = int(count * 0.70)
        added = 0

        for i in range(len(y)):
            if y[i] == c:
                if added < train_count:
                    X_train.append(X[i])
                    y_train.append(y[i])
                    added += 1
                else:
                    X_test.append(X[i])
                    y_test.append(y[i])

    return X_train, X_test, y_train, y_test

# Calculate Euclidean distance
def euclidean_distance(a, b):
    total = 0.0
    for i in range(len(a)):
        difference = a[i] - b[i]
        total += difference * difference

    return math.sqrt(total)

# Predict class using K nearest neighbors
def predict_knn(X_train, y_train, sample, k):
    distances = []

    for i in range(len(X_train)):
        distance = euclidean_distance(X_train[i], sample)
        distances.append([distance, y_train[i]])

    # Sort distances manually using bubble sort
    for i in range(len(distances)):
        for j in range(0, len(distances) - i - 1):
            if distances[j][0] > distances[j + 1][0]:
                distances[j], distances[j + 1] = (
                    distances[j + 1],
                    distances[j]
                )

    # Count votes of nearest neighbors
    votes = [0, 0, 0]

    for i in range(k):
        label = distances[i][1]
        votes[label] += 1

    # Return class with maximum votes
    best_class = 0

    for c in range(1, len(votes)):
        if votes[c] > votes[best_class]:
            best_class = c

    return best_class

# Calculate accuracy manually
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / len(actual)

# Main program
X_train, X_test, y_train, y_test = split_data(X, y)
k = 7
predictions = []

for sample in X_test:
    prediction = predict_knn(X_train, y_train, sample, k)
    predictions.append(prediction)

accuracy = calculate_accuracy(y_test, predictions)

print("KNN Classification")
print("K value:", k)
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Accuracy (%):", accuracy * 100)