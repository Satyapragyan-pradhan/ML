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

# 2D Haar DWT 
def haar_dwt(image):
    rows = len (image)
    cols = len (image [0] )

    # First perform horizontal transformation
    horizontal_low = []
    horizontal_high = []
    for i in range (rows):
        low_row = []
        high_row = []
        for j in range (0, cols, 2):
            a = image[i][j]
            b = image[i][j + 1]

            low = (a + b) / 2
            high = (a - b) / 2
            low_row.append(low)
            high_row.append(high)

        horizontal_low.append(low_row)
        horizontal_high.append(high_row)


    # Now perform vertical transformation
    LL = []
    LH = []
    HL = []
    HH = []

    for i in range(0, rows, 2):
        ll_row = []
        lh_row = []
        hl_row = []
        hh_row = []
        for j in range (cols // 2) :
            a = horizontal_low[i][j]
            b = horizontal_low[i+1][j]
            c = horizontal_high[i][j]
            d = horizontal_high[i+1][j]

            ll_row.append((a+b) / 2)
            lh_row.append((a-b) / 2)
            hl_row.append((c+d) / 2)
            hh_row.append((c-d) / 2)

        LL. append(ll_row)
        LH. append(lh_row)
        HL. append(hl_row)
        HH. append(hh_row)

    return LL, LH, HL, HH

# main
image = read_one_image (IMAGE_FILE, IMAGE_INDEX)
LL, LH, HL, HH = haar_dwt (image)

print ("MNIST DWT IMPLEMENTATION")
print ("Original Image Size : 28 x 28")
print ("Reduced Image Size : 14 x 14")

print ("\nDWT Subbands : ")
print ("LL (Low-Low) : 14 x 14")
print ("LH (Low-High) : 14 x 14")
print ("HL (High-Low) : 14 x 14")
print ("HH (High-High) : 14 x 14")

# Display Original and DWT Outputs 
plt.figure(figsize=(10, 7))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original 28x28")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(LL, cmap="gray")
plt.title("LL - Low Frequency")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(LH, cmap="gray")
plt.title("LH - Horizontal Detail")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(HL, cmap="gray")
plt.title("HL - Vertical Detail")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(HH, cmap="gray")
plt.title("HH - Diagonal Detail")
plt.axis("off")

plt.subplots_adjust(wspace=0.4,hspace=0.5)
plt.show()

