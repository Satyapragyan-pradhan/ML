import csv
from sklearn.svm import SVC

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

# Perform K-Fold Cross-Validation manually
def k_fold_split(X, y, k):
    folds = []

    for i in range(k):
        folds.append([[], []])

    for c in [0, 1, 2]:
        class_indices = []
        for i in range(len(y)):
            if y[i] == c:
                class_indices.append(i)

        for i in range(len(class_indices)):
            fold_index = i % k
            index = class_indices[i]
            folds[fold_index][0].append(X[index])
            folds[fold_index][1].append(y[index])

    return folds

# Calculate accuracy manually
def accuracy_score(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1

    return correct / len(actual)


# Main program
k = 5
folds = k_fold_split(X, y, k)
total_accuracy = 0

for fold in range(k):
    X_train, X_test = [], []
    y_train, y_test = [], []

    # Use one fold for testing and remaining folds for training
    for i in range(k):
        if i == fold:
            X_test += folds[i][0]
            y_test += folds[i][1]
        else:
            X_train += folds[i][0]
            y_train += folds[i][1]

    # Train SVM model
    model = SVC(kernel="linear")
    model.fit(X_train, y_train)

    # Predict test samples
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    total_accuracy += accuracy
    print("Fold", fold + 1, "Accuracy:", accuracy * 100, "%")

# Calculate average accuracy
average_accuracy = total_accuracy / k
print("\nAverage Accuracy:", average_accuracy * 100, "%")
