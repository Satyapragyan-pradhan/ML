import csv
from sklearn.svm import SVC
import matplotlib.pyplot as plt

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

# Calculate confusion matrix manually
def confusion_matrix(actual, predicted, classes):
    matrix = []

    for i in range(len(classes)):
        row = []
        for j in range(len(classes)):
            row.append(0)
        matrix.append(row)

    for i in range(len(actual)):
        matrix[actual[i]][predicted[i]] += 1

    return matrix

# Calculate evaluation metrics manually
def evaluate(actual, predicted, classes):
    matrix = confusion_matrix(actual, predicted, classes)

    total = len(actual)
    correct = 0
    for i in range(len(classes)):
        correct += matrix[i][i]

    accuracy = correct / total
    print("\nConfusion Matrix:")
    for row in matrix:
        print(row)

    print("\nClass-wise Evaluation:")
    for c in classes:
        TP = matrix[c][c]
        FP = 0
        FN = 0
        TN = 0

        for i in classes:
            if i != c:
                FP += matrix[i][c]
                FN += matrix[c][i]

        TN = total - TP - FP - FN
        sensitivity = TP / (TP + FN) if TP + FN else 0
        specificity = TN / (TN + FP) if TN + FP else 0
        precision = TP / (TP + FP) if TP + FP else 0

        if precision + sensitivity:
            f1 = 2 * precision * sensitivity / (
                precision + sensitivity
            )
        else:
            f1 = 0

        print("\nClass:", c)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)
        print("Precision:", precision)
        print("F1-score:", f1)

    print("\nOverall Accuracy:", accuracy)
    print("Accuracy (%):", accuracy * 100)

    return matrix

# Main program
X_train, X_test, y_train, y_test = split_data(X, y)
model = SVC(kernel="linear")
model.fit(X_train, y_train)

predictions = model.predict(X_test).tolist()
classes = [0, 1, 2]
matrix = evaluate(y_test, predictions, classes)

# ROC curve using one-vs-rest approach
def roc_points(actual, scores, positive_class):
    points = []
    thresholds = []

    for score_row in scores:
        for score in score_row:
            found = False
            for threshold in thresholds:
                if threshold == score:
                    found = True
                    break

            if not found:
                thresholds.append(score)

    thresholds.append(float("inf"))
    thresholds.append(float("-inf"))

    for threshold in thresholds:
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for i in range(len(actual)):
            predicted_positive = (
                scores[i][positive_class] >= threshold
            )
            actual_positive = actual[i] == positive_class

            if predicted_positive and actual_positive:
                TP += 1
            elif predicted_positive and not actual_positive:
                FP += 1
            elif not predicted_positive and actual_positive:
                FN += 1
            else:
                TN += 1

        TPR = TP / (TP + FN) if TP + FN else 0
        FPR = FP / (FP + TN) if FP + TN else 0

        points.append([FPR, TPR])

    # Sort points by false-positive rate
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if points[j][0] < points[i][0]:
                points[i], points[j] = points[j], points[i]

    return points


# Get decision scores from SVM
scores = model.decision_function(X_test).tolist()
for c in classes:
    points = roc_points(y_test, scores, c)
    fpr = []
    tpr = []
    for point in points:
        fpr.append(point[0])
        tpr.append(point[1])

    plt.plot(fpr, tpr, label="Class " + str(c))

plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve - Iris Dataset")
plt.legend()
plt.grid(True)
plt.show()