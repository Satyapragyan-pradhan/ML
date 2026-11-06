import math

# Calculate dot product
def dot_product(A, B):
    total = 0.0
    for i in range(len(A)):
        total += A[i] * B[i]
    return total

# Calculate vector magnitude
def magnitude(A):
    total = 0.0
    for value in A:
        total += value * value

    return math.sqrt(total)

# Calculate cosine similarity
def cosine_similarity(A, B):
    mag_A = magnitude(A)
    mag_B = magnitude(B)
    if mag_A == 0 or mag_B == 0:
        return 0.0

    return dot_product(A, B) / (mag_A * mag_B)

# Example
A = [1, 2, 3]
B = [2, 4, 6]
C = [3, 1, 2]

print("Similarity A and B:", cosine_similarity(A, B))
print("Similarity A and C:", cosine_similarity(A, C))