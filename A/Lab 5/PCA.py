import struct
import matplotlib.pyplot as plt

IMAGE_FILE = "train-images.idx3-ubyte"
COMPONENTS = 14  # 28 -> 14 dimensions 
IMAGE_INDEX = 0  # First MNIST image

# Read one MNIST image 
def read_one_image (filename, index) :
    f = open (filename, "rb")
    magic, count, rows, cols = struct. unpack (">IIII", f. read (16) )

    f. seek (16 + index * rows * cols)
    image = []
    for i in range (rows) :
        row = []
        for j in range (cols) :
            pixel = f.read(1)[0] / 255.0
            row. append (pixel)
        image.append (row)

    f.close ()
    return image

#  Mean of each column 
def column_mean (matrix) :
    rows = len (matrix)
    cols = len (matrix [0] )
    mean = [0.0] * cols

    for j in range (cols) :
        s = 0.0
        for i in range (rows) :
            s += matrix [i] [j]
        mean[j] = s / rows

    return mean

# Center matrix 
def center(matrix, mean):     
    rows = len(matrix)     
    cols = len(matrix[0]) 
         
    result = []      
    for i in range(rows):         
        row = []         
        for j in range(cols):             
            row.append(matrix[i][j] - mean[j])         
        result.append(row)      
    return result 

# Covariance Matrix (28 x 28) 
def covariance (matrix) :
    rows = len (matrix)
    cols = len (matrix [0] )

    cov = []
    for i in range (cols) :
        cov_row = []
        for j in range (cols) :
            s = 0.0
            for k in range (rows) :
                s += matrix [k] [i] * matrix [k] [j]

            cov_row. append (s / (rows - 1) )
        cov. append (cov_row)
    return cov

# Dot Product 
def dot(a, b):     
    s = 0.0     
    for i in range(len(a)):         
        s += a[i] * b[i]     
    return s 

# Matrix x Vector 
def mat_vec (matrix, vector) :
    result = []
    for row in matrix:
        result.append(dot(row, vector))

    return result

# Normalize vector 
def normalize (v) :
    s = dot (v, v) ** 0.5

    if s == 0:
        return v

    return [x / s for x in v]

# Power Iteration
def power_iteration (matrix, previous) :
    n = len(matrix)
    v = [1.0] * n
    for i in range(40) :
        for p in previous:
            proj = dot(v, p)
            for i in range(n):
                v[i] -= proj * p[i]

        v = mat_vec (matrix, v)
        v = normalize (v)

    eigenvalue = dot (v, mat_vec (matrix, v))
    return v, eigenvalue

# Compute PCA Basis 
def pca_basis (cov, k) :
    basis = []
    values = []
    for i in range(k):
        vec, val = power_iteration(cov, basis)
        basis.append (vec)
        values. append (val)

    return basis, values

#  Project Image 
def project (centered, basis) :
    rows = len (centered)
    reduced = []

    for i in range (rows) :
        coeff = []
        for b in basis:
            coeff.append (dot (centered[i], b))
        reduced. append (coeff)

    return reduced

# Reconstruct Image
def reconstruct (reduced, basis, mean) :
    rows = len (reduced)
    reconstructed = []
    for i in range (rows) :
        row = []
        for j in range (len (mean) ) :
            value = mean [j]
            for k in range (len (basis)):
                value += reduced [i] [k] * basis [k] [j]
            row.append(value)
        reconstructed.append(row)
    return reconstructed

# main
image = read_one_image(IMAGE_FILE, IMAGE_INDEX)

mean = column_mean (image)
centered = center (image, mean)
cov = covariance (centered)

basis, eigenvalues = pca_basis (cov, COMPONENTS)

reduced = project (centered, basis)
reconstructed = reconstruct (reduced, basis, mean)

# Terminal Output
print ("MNIST PCA IMPLEMENTATION")
print ("Original Image Size : 28 x 28")
print ("Reduced Dimensions : 28 x", COMPONENTS)

total = sum (eigenvalues)
print ("\nTop 5 Eigenvalues")
for i in range(5) :
    print ("PC", i + 1, ":", round (eigenvalues [i], 4) )

print ("\nSum of Eigenvalues of First", COMPONENTS, "Components:", round (total, 4))

# Display Images
plt.figure (figsize=(12, 4) )
plt.subplot (1, 3, 1)
plt. imshow (image, cmap="gray")

plt.title("Original 28x28")
plt.axis ("off")

plt.subplot (1, 3, 2)
plt. imshow (reduced, cmap="gray")
plt. title ("Reduced 28x14")
plt.axis ("off")

plt.subplot (1, 3, 3)
plt.imshow (reconstructed, cmap="gray")
plt.title ("Reconstructed 28x28")
plt. axis ("off")

plt.subplots_adjust (wspace=0.5)
plt.show ()

