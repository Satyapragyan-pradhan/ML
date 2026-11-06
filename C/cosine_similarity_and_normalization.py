import numpy as np
import pandas as pd

# ================= 1. NORMALIZATION FUNCTIONS =================
def min_max_normalize(data):
    """Scales values to the range [0, 1]."""
    data = np.array(data, dtype=float)
    min_val, max_val = data.min(), data.max()
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def z_score_standardize(data):
    """Standardizes values (Mean = 0, Std Dev = 1), population std."""
    data = np.array(data, dtype=float)
    std_dev = data.std()          # ddof=0, same as the original code
    if std_dev == 0:
        return np.zeros_like(data)
    return (data - data.mean()) / std_dev

def l2_normalize(vector):
    """Rescales a vector so that its L2 norm equals 1."""
    vector = np.array(vector, dtype=float)
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return np.zeros_like(vector)
    return vector / magnitude


# ================= 2. COSINE SIMILARITY FUNCTIONS =================
def cosine_similarity(v1, v2):
    """Cosine similarity between two 1D vectors."""
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    mag1, mag2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return np.dot(v1, v2) / (mag1 * mag2)

def pairwise_cosine_similarity(matrix):
    """Pairwise cosine similarity matrix for a dataset (rows = samples)."""
    matrix = np.array(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return np.round(normalized @ normalized.T, 4)


# ================= 3. DEMONSTRATION =================
raw_data = [10, 20, 30, 40, 50]

print("=== DATA NORMALIZATION ===")
print("Original Data:        ", raw_data)
print("Min-Max Normalized:   ", np.round(min_max_normalize(raw_data), 4).tolist())
print("Z-Score Standardized: ", np.round(z_score_standardize(raw_data), 4).tolist())
print("L2 Normalized Vector: ", np.round(l2_normalize(raw_data), 4).tolist())

print("\n" + "=" * 50 + "\n")

vector_A = [3, 45, 7, 2]
vector_B = [2, 54, 13, 15]

print("Vector A:", vector_A)
print("Vector B:", vector_B)
sim_score = cosine_similarity(vector_A, vector_B)
print(f"Cosine Similarity Score: {sim_score:.4f}")

print("\n" + "=" * 50 + "\n")

dataset_matrix = [
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],   # collinear with row 0 (similarity = 1.0)
    [5.0, 0.0, 1.0]
]

print("Dataset Matrix:")
for row in dataset_matrix:
    print(row)

print("\nPairwise Cosine Similarity Matrix:")
sim_matrix = pairwise_cosine_similarity(dataset_matrix)
print(pd.DataFrame(sim_matrix))