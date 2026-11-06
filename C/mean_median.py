import numpy as np
import pandas as pd

# ---------------- A.1 : Mean, Median, Mode, Std, Variance, Z-Score ----------------
print("A.1. STATISTICAL MEASURES & Z-SCORES")

data = np.array([12, 15, 18, 15, 22, 25, 28, 30, 15, 35, 40, 48, 55])

mean_val = np.mean(data)
median_val = np.median(data)
mode_val = pd.Series(data).mode().values[0]
std_dev = np.std(data, ddof=1)       # sample standard deviation
variance = np.var(data, ddof=1)      # sample variance
z_scores = (data - np.mean(data)) / np.std(data)   # same as scipy zscore (ddof=0)

print(f"Data Series: {data}")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Mode: {mode_val}")
print(f"Standard Deviation: {std_dev:.2f}")
print(f"Variance: {variance:.2f}\n")

df_zscore = pd.DataFrame({'Value': data, 'Z-Score': np.round(z_scores, 2)})
print("Z-Scores for Data Points:")
print(df_zscore.to_string(index=False))


# ---------------- A.2 : Range, Q1, Q2, Q3, IQR ----------------
print("\nA.2. QUANTILES AND IQR")

arr = np.array([10, 24, 28, 32, 35, 38, 42, 45, 50, 55, 60, 85])

data_range = np.max(arr) - np.min(arr)
q1 = np.percentile(arr, 25)
q2 = np.percentile(arr, 50)    # median
q3 = np.percentile(arr, 75)
iqr = q3 - q1

print(f"Array: {arr}")
print(f"Range: {data_range}")
print(f"Q1 (25th Percentile): {q1:.2f}")
print(f"Q2 (50th Percentile/Med): {q2:.2f}")
print(f"Q3 (75th Percentile): {q3:.2f}")
print(f"IQR (Interquartile Range): {iqr:.2f}")