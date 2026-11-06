from sklearn.datasets import load_iris
import math

# Load Iris dataset
iris = load_iris()
X = iris.data.tolist()
y = iris.target.tolist()
feature_names = iris.feature_names

# Calculate Chi-Square score for one feature
def chi_square_feature(X, y, feature):
    observed = [[0, 0, 0], [0, 0, 0]]
    total = len(y)

    # Find minimum and maximum
    low = X[0][feature]
    high = X[0][feature]
    for row in X:
        if row[feature] < low:
            low = row[feature]
        if row[feature] > high:
            high = row[feature]

    threshold = (low + high) / 2

    # Build observed frequency table
    for i in range(total):
        if X[i][feature] <= threshold:
            category = 0
        else:
            category = 1
        observed[category][y[i]] += 1

    row_totals = [0, 0]
    col_totals = [0, 0, 0]
    for i in range(2):
        for j in range(3):
            row_totals[i] += observed[i][j]
            col_totals[j] += observed[i][j]

    # Calculate Chi-Square statistic
    score = 0.0
    for i in range(2):
        for j in range(3):
            expected = (row_totals[i] * col_totals[j] / total)
            if expected != 0:
                difference = observed[i][j] - expected
                score += difference * difference / expected

    return score

# Main program
print("Chi-Square Feature Selection")
scores = []
for feature in range(len(feature_names)):
    score = chi_square_feature(X, y, feature)
    scores.append([score, feature])
    print(feature_names[feature], ":", round(score, 4))

# Sort scores manually in descending order
for i in range(len(scores)):
    for j in range(i + 1, len(scores)):
        if scores[j][0] > scores[i][0]:
            scores[i], scores[j] = scores[j], scores[i]

print("\nFeature Ranking:")
for i in range(len(scores)):
    feature = scores[i][1]
    print(i+1, feature_names[feature], "Score:", round(scores[i][0], 4) )