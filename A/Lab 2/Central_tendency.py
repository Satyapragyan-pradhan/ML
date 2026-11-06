from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.tolist()
names = iris.feature_names

# Mean
def mean(data):
    total = 0
    for x in data:
        total += x
    return total / len(data)

# Median
def median(data):
    arr = data.copy()
    n = len(arr)

    # Manual sorting
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]

    if n % 2 == 1:
        return arr[n // 2]
    return (arr[n // 2 - 1] + arr[n // 2]) / 2

# Mode
def mode(data):
    frequency = []

    for x in data:
        count = 0
        for y in data:
            if x == y:
                count += 1

        found = False
        for item in frequency:
            if item[0] == x:
                found = True
                break

        if not found:
            frequency.append([x, count])

    max_count = 0
    result = []
    for x, count in frequency:
        if count > max_count:
            max_count = count
            result = [x]
        elif count == max_count:
            result.append(x)

    return result

print("Measures of Central Tendency")
for col in range(len(names)):
    data = [row[col] for row in X]

    print(names[col])
    print("Mean   :", round(mean(data), 3))
    print("Median :", round(median(data), 3))
    print("Mode   :", mode(data))
    print()









names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# A small sample representing rows of measurements
X = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1],
]


# Mean
def mean(data):
    total = 0
    for x in data:
        total += x
    return total / len(data)


# Median
def median(data):
    arr = data.copy()
    n = len(arr)

    # Manual sorting
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]

    if n % 2 == 1:
        return arr[n // 2]
    return (arr[n // 2 - 1] + arr[n // 2]) / 2


# Mode
def mode(data):
    frequency = []

    for x in data:
        count = 0
        for y in data:
            if x == y:
                count += 1

        found = False
        for item in frequency:
            if item[0] == x:
                found = True
                break

        if not found:
            frequency.append([x, count])

    max_count = 0
    result = []
    for x, count in frequency:
        if count > max_count:
            max_count = count
            result = [x]
        elif count == max_count:
            result.append(x)

    return result


print("Measures of Central Tendency")
for col in range(len(names)):
    data = [row[col] for row in X]

    print(names[col])
    print("Mean   :", round(mean(data), 3))
    print("Median :", round(median(data), 3))
    print("Mode   :", mode(data))
    print()
