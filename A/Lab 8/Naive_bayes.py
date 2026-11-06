from sklearn.datasets import load_iris
import math

iris = load_iris()

X = iris.data.tolist()
y = iris.target.tolist()
feature_names = iris.feature_names

# Get unique class labels
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

# Split data manually into 70% training and 30% testing
def split_data(X, y):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    classes = get_classes(y)
    for current_class in classes:
        class_count = 0
        for label in y:
            if label == current_class:
                class_count += 1

        test_count = int(class_count * 0.30)
        train_count = class_count - test_count

        current_train = 0
        current_test = 0

        for i in range(len(y)):
            if y[i] == current_class:

                if current_train < train_count:
                    X_train.append(X[i])
                    y_train.append(y[i])
                    current_train += 1

                elif current_test < test_count:
                    X_test.append(X[i])
                    y_test.append(y[i])
                    current_test += 1

    return X_train, X_test, y_train, y_test

# Calculate mean
def calculate_mean(values):
    total = 0.0
    for value in values:
        total += value

    return total / len(values)

# Calculate variance
def calculate_variance(values):
    mean = calculate_mean(values)
    total = 0.0
    for value in values:
        difference = value - mean
        total += difference * difference

    return total / len(values)

# Calculate Gaussian probability
def gaussian_probability(x, mean, variance):
    if variance == 0:
        variance = 0.000000001

    exponent = math.exp(
        -((x - mean) ** 2) /
        (2 * variance)
    )

    denominator = math.sqrt(2 * math.pi * variance)
    return exponent / denominator

# Calculate mean and variance for every class
def calculate_statistics(X_train, y_train, classes):
    statistics = {}
    for current_class in classes:
        class_data = []
        for i in range(len(y_train)):
            if y_train[i] == current_class:
                class_data.append(X_train[i])

        feature_statistics = []
        for feature in range(4):
            values = []
            for row in class_data:
                values.append(row[feature])

            mean = calculate_mean(values)
            variance = calculate_variance(values)

            feature_statistics.append(
                [mean, variance]
            )

        statistics[current_class] = (
            feature_statistics
        )

    return statistics

# Calculate prior probability of each class
def calculate_priors(y_train, classes):
    priors = {}
    for current_class in classes:
        count = 0
        for label in y_train:
            if label == current_class:
                count += 1

        priors[current_class] = (count / len(y_train))

    return priors

# Predict class for one sample
def predict_sample(
    sample,
    classes,
    statistics,
    priors
):

    best_class = None
    best_probability = -1
    for current_class in classes:
        probability = priors[current_class]
        for feature in range(4):
            mean = statistics[current_class][feature][0]
            variance = statistics[current_class][feature][1]
            probability *= gaussian_probability(
                sample[feature],
                mean,
                variance
            )

        if probability > best_probability:
            best_probability = probability
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
statistics = calculate_statistics(X_train,y_train,classes)

priors = calculate_priors(y_train,classes)
predictions = []
for sample in X_test:
    prediction = predict_sample(
        sample,
        classes,
        statistics,
        priors
    )

    predictions.append(prediction)

accuracy = calculate_accuracy(y_test,predictions)

print("NAIVE BAYES CLASSIFICATION")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("\nClass Priors:")

for current_class in classes:
    print(
        "Class",
        current_class,
        "->",
        priors[current_class]
    )

print("Accuracy (%):",accuracy * 100)