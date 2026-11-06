from sklearn.datasets import load_iris
from sklearn.svm import SVC

iris = load_iris()
X = iris.data.tolist()
y = iris.target.tolist()
feature_names = iris.feature_names

# Split data into training and testing sets
def split_data(X, y):
    X_train, X_test = [], []
    y_train, y_test = [], []

    classes = []

    for label in y:
        found = False
        for c in classes:
            if c == label:
                found = True
                break

        if not found:
            classes.append(label)

    for c in classes:
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

# Calculate accuracy manually
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / len(actual)

# Main program
X_train, X_test, y_train, y_test = split_data(X, y)

# Train the SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict test samples
predictions = model.predict(X_test)
accuracy = calculate_accuracy(y_test, predictions)

print("SVM Classification")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)
