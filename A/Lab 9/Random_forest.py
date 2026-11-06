import csv
import random

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


# Calculate Gini impurity
def gini(labels):
    if len(labels) == 0:
        return 0
    impurity = 1.0
    for c in [0, 1, 2]:
        count = 0
        for label in labels:
            if label == c:
                count += 1

        probability = count / len(labels)
        impurity -= probability * probability

    return impurity


# Find majority class
def majority_class(labels):
    best_class = labels[0]
    best_count = 0
    for c in [0, 1, 2]:
        count = 0
        for label in labels:
            if label == c:
                count += 1
        if count > best_count:
            best_count = count
            best_class = c

    return best_class


# Build one decision tree
def build_tree(X, y, depth=0, max_depth=5):
    if len(y) == 0:
        return {"class": 0}

    classes = []
    for label in y:
        found = False
        for c in classes:
            if c == label:
                found = True
                break

        if not found:
            classes.append(label)

    if len(classes) == 1 or depth >= max_depth:
        return {"class": majority_class(y)}

    best_gini = gini(y)
    best_feature = -1
    best_threshold = 0
    best_left_X, best_left_y = [], []
    best_right_X, best_right_y = [], []

    # Randomly select two features at each node
    features = [0, 1, 2, 3]
    for i in range(len(features) - 1, 0, -1):
        j = random.randint(0, i)
        features[i], features[j] = features[j], features[i]

    selected_features = features[:2]

    for feature in selected_features:
        values = []
        for row in X:
            values.append(row[feature])

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                threshold = (values[i] + values[j]) / 2

                left_X, right_X = [], []
                left_y, right_y = [], []

                for k in range(len(X)):
                    if X[k][feature] <= threshold:
                        left_X.append(X[k])
                        left_y.append(y[k])
                    else:
                        right_X.append(X[k])
                        right_y.append(y[k])

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                weighted_gini = (
                    len(left_y) * gini(left_y)
                    + len(right_y) * gini(right_y)
                ) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold
                    best_left_X, best_left_y = left_X, left_y
                    best_right_X, best_right_y = right_X, right_y

    if best_feature == -1:
        return {"class": majority_class(y)}

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": build_tree(
            best_left_X, best_left_y, depth + 1, max_depth
        ),
        "right": build_tree(
            best_right_X, best_right_y, depth + 1, max_depth
        )
    }

# Predict using one tree
def predict_tree(tree, sample):
    if "class" in tree:
        return tree["class"]
    feature = tree["feature"]
    threshold = tree["threshold"]

    if sample[feature] <= threshold:
        return predict_tree(tree["left"], sample)
    return predict_tree(tree["right"], sample)

# Train Random Forest using bootstrap samples
def train_forest(X, y, tree_count):
    forest = []
    n = len(X)

    for t in range(tree_count):
        sample_X, sample_y = [], []
        for i in range(n):
            index = random.randint(0, n - 1)
            sample_X.append(X[index])
            sample_y.append(y[index])

        tree = build_tree(sample_X, sample_y)
        forest.append(tree)

    return forest

# Predict using majority voting
def predict_forest(forest, sample):
    votes = [0, 0, 0]
    for tree in forest:
        prediction = predict_tree(tree, sample)
        votes[prediction] += 1

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
random.seed(42)
X_train, X_test, y_train, y_test = split_data(X, y)
forest = train_forest(X_train, y_train, tree_count=10)
predictions = []

for sample in X_test:
    predictions.append(predict_forest(forest, sample))
accuracy = calculate_accuracy(y_test, predictions)

print("Random Forest Classification")
print("Number of Trees:", len(forest))
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))
print("Accuracy (%):", accuracy * 100)