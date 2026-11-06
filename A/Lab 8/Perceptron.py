from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.tolist()
y = iris.target.tolist()
feature_names = iris.feature_names

# Get unique classes
def get_classes(y):
    classes = []
    for label in y:
        found = False
        for current in classes:
            if label == current:
                found = True
                break
        if not found:
            classes.append(label)

    return classes

# Split data manually into training and testing sets
def split_data(X, y):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    classes = get_classes(y)
    for current_class in classes:
        count = 0
        for label in y:
            if label == current_class:
                count += 1

        train_count = int(count * 0.70)
        train_added = 0
        test_added = 0

        for i in range(len(y)):
            if y[i] == current_class:
                if train_added < train_count:
                    X_train.append(X[i])
                    y_train.append(y[i])
                    train_added += 1
                else:
                    X_test.append(X[i])
                    y_test.append(y[i])
                    test_added += 1

    return X_train, X_test, y_train, y_test

# Calculate dot product
def dot_product(x, weights):
    total = 0.0
    for i in range(len(x)):
        total += x[i] * weights[i]

    return total

# Train perceptron using one-vs-rest approach
def train_perceptron(X_train, y_train, target_class):
    weights = [0.0, 0.0, 0.0, 0.0]
    bias = 0.0
    learning_rate = 0.01
    epochs = 100

    for epoch in range(epochs):
        errors = 0
        for i in range(len(X_train)):
            if y_train[i] == target_class:
                actual = 1
            else:
                actual = 0

            value = dot_product(
                X_train[i],
                weights
            ) + bias

            if value >= 0:
                predicted = 1
            else:
                predicted = 0

            error = actual - predicted

            if error != 0:
                for j in range(len(weights)):
                    weights[j] += learning_rate * error * X_train[i][j]

                bias += learning_rate * error
                errors += 1

        if errors == 0:
            break

    return weights, bias

# Predict probability/class score for one class
def predict_score(sample, weights, bias):
    return dot_product(sample, weights) + bias

# Predict class using all binary perceptrons
def predict_sample(sample, models, classes):
    best_class = classes[0]
    best_score = predict_score(
        sample,
        models[classes[0]][0],
        models[classes[0]][1]
    )

    for i in range(1, len(classes)):
        current_class = classes[i]
        score = predict_score(
            sample,
            models[current_class][0],
            models[current_class][1]
        )

        if score > best_score:
            best_score = score
            best_class = current_class

    return best_class

# Calculate accuracy manually
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / len(actual)

# MAIN
X_train, X_test, y_train, y_test = split_data(X,y)
classes = get_classes(y_train)
models = {}

# Train one perceptron for every class
for current_class in classes:
    weights, bias = train_perceptron(
        X_train,
        y_train,
        current_class
    )

    models[current_class] = (weights,bias)

# Predict test data
predictions = []
for sample in X_test:
    prediction = predict_sample(
        sample,
        models,
        classes
    )

    predictions.append(prediction)

# Calculate accuracy
accuracy = calculate_accuracy(y_test,predictions)

print("PERCEPTRON CLASSIFICATION")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Accuracy (%):", accuracy * 100)