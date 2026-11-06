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

# Calculate squared Euclidean distance
def squared_distance(a, b):
    total = 0.0
    for i in range(len(a)):
        difference = a[i] - b[i]
        total += difference * difference

    return total

# Calculate Gaussian kernel value
def gaussian_kernel(distance_squared, spread):
    return math.exp(-distance_squared / (2 * spread * spread))

# Predict class using PNN
def predict_pnn(X_train, y_train, sample, classes, spread):
    class_scores = []
    for c in classes:
        score = 0.0
        count = 0
        for i in range(len(X_train)):
            if y_train[i] == c:
                distance = squared_distance(sample, X_train[i])
                score += gaussian_kernel(distance, spread)
                count += 1
        if count > 0:
            score = score / count
        class_scores.append(score)

    best_class = 0

    for i in range(1, len(classes)):
        if class_scores[i] > class_scores[best_class]:
            best_class = i

    return classes[best_class]

# Calculate accuracy manually
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / len(actual)


# Main program
X_train, X_test, y_train, y_test = split_data(X, y)
classes = [0, 1, 2]
spread = 0.5

predictions = []
for sample in X_test:
    prediction = predict_pnn(X_train, y_train, sample, classes, spread)
    predictions.append(prediction)

accuracy = calculate_accuracy(y_test, predictions)

print("Probabilistic Neural Network")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Spread:", spread)
print("Accuracy (%):", accuracy * 100)