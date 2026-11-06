from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data.tolist()
names = iris.feature_names

def quartiles(data):
    arr = data.copy()
    n = len(arr)

    # Manual sorting
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] > arr[j]:
                arr[i], arr[j] = arr[j], arr[i]

    # Median of a sorted list
    def median(values):
        m = len(values)
        if m % 2 == 1:
            return values[m // 2]
        return (values[m // 2 - 1] + values[m // 2]) / 2

    q2 = median(arr)

    # Split data into lower and upper halves
    if n % 2 == 0:
        lower = arr[:n // 2]
        upper = arr[n // 2:]
    else:
        lower = arr[:n // 2]
        upper = arr[n // 2 + 1:]

    q1 = median(lower)
    q3 = median(upper)

    return q1, q2, q3

print("Quartile Analysis")
for col in range(len(names)):
    data = [row[col] for row in X]

    q1, q2, q3 = quartiles(data)
    iqr = q3 - q1

    print(names[col])
    print("Q1  :", round(q1, 3))
    print("Q2  :", round(q2, 3))
    print("Q3  :", round(q3, 3))
    print("IQR :", round(iqr, 3))
    print()