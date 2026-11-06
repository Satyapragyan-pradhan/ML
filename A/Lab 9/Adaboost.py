import csv
import math
import os

X = []
y = []

label_mapping = {}
label_counter = 0

with open("Iris.csv", mode='r') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    
    for row in reader:
        if not row or len(row) < 6:  # Skip empty or corrupted lines
            continue
            
        X.append([float(val) for val in row[1:5]])
        raw_label = row[5].strip()
        if raw_label not in label_mapping:
            label_mapping[raw_label] = label_counter
            label_counter += 1
            
        y.append(label_mapping[raw_label])

# Split data manually into training and testing sets
def split_data(X, y):
    X_train, X_test = [], []
    y_train, y_test = [], []

    unique_classes = sorted(list(set(y)))
    for c in unique_classes:
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

# Find the best decision stump using weighted error
def find_stump(X, y, weights):
    best_error = float("inf")
    best_stump = {}

    for feature in range(len(X[0])):
        values = []
        for row in X:
            values.append(row[feature])
        
        # Sort values to evaluate thresholds sequentially between points
        unique_values = sorted(list(set(values)))
        
        for i in range(len(unique_values) - 1):
            threshold = (unique_values[i] + unique_values[i + 1]) / 2
            for polarity in [1, -1]:
                error = 0
                for k in range(len(X)):
                    if polarity == 1:
                        prediction = 0 if X[k][feature] <= threshold else 1
                    else:
                        prediction = 1 if X[k][feature] <= threshold else 0

                    if prediction != y[k]:
                        error += weights[k]

                if error < best_error:
                    best_error = error
                    best_stump = {
                        "feature": feature,
                        "threshold": threshold,
                        "polarity": polarity
                    }

    return best_stump, best_error

# Predict using one decision stump
def predict_stump(stump, sample):
    feature = stump["feature"]
    threshold = stump["threshold"]
    if stump["polarity"] == 1:
        return 0 if sample[feature] <= threshold else 1

    return 1 if sample[feature] <= threshold else 0

# Train AdaBoost using binary labels
def train_adaboost(X_train, y_train, rounds=10):
    models = []
    unique_classes = sorted(list(set(y_train)))
    for positive_class in unique_classes:
        binary_y = []
        for label in y_train:
            binary_y.append(1 if label == positive_class else 0)

        n = len(binary_y)
        weights = []
        for i in range(n):
            weights.append(1 / n)

        stumps = []

        for iteration in range(rounds):
            stump, error = find_stump(X_train, binary_y, weights)

            if error >= 0.5:
                break
            if error <= 0:
                error = 0.000000001

            alpha = 0.5 * math.log((1 - error) / error)
            for i in range(n):
                prediction = predict_stump(stump, X_train[i])
                if prediction == binary_y[i]:
                    weights[i] *= math.exp(-alpha)
                else:
                    weights[i] *= math.exp(alpha)

            total_weight = 0
            for weight in weights:
                total_weight += weight
            for i in range(n):
                weights[i] /= total_weight

            stumps.append([stump, alpha])
        models.append((positive_class, stumps))

    return models

# Predict class using one-vs-rest AdaBoost models
def predict_adaboost(models, sample):
    best_class = None
    best_score = float("-inf")

    for positive_class, stumps in models:
        score = 0
        for stump, alpha in stumps:
            prediction = predict_stump(stump, sample)
            if prediction == 1:
                score += alpha
            else:
                score -= alpha

        if score > best_score:
            best_score = score
            best_class = positive_class

    return best_class

# Calculate accuracy manually
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / len(actual)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_adaboost(X_train, y_train, rounds=10)
    
    predictions = []
    for sample in X_test:
        predictions.append(predict_adaboost(models, sample))

    accuracy = calculate_accuracy(y_test, predictions)

    print("\nAdaBoost Classification Results")
    print("-" * 30)
    print("Classes Detected :", list(label_mapping.keys()))
    print("Training Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))
    print(f"Accuracy (%)     : {accuracy * 100:.2f}%")
