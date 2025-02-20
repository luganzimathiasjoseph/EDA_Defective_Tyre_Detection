import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.feature import local_binary_pattern

# Paths
good_tires_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of defective and good condition tyres\good"
defective_tires_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of defective and good condition tyres\defective"

# Select one good and one defective tire image
good_img_path = os.path.join(good_tires_dir, os.listdir(good_tires_dir)[0])
defective_img_path = os.path.join(defective_tires_dir, os.listdir(defective_tires_dir)[0])

def process_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))

    # LBP (Local Binary Pattern)
    lbp = local_binary_pattern(image, P=8, R=1, method="uniform")

    # Wavelet Transform (DWT)
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2  # Extract low and high-frequency components

    return image, lbp, LL, LH, HL, HH

# Process images
good_image, good_lbp, good_LL, good_LH, good_HL, good_HH = process_image(good_img_path)
defective_image, defective_lbp, defective_LL, defective_LH, defective_HL, defective_HH = process_image(defective_img_path)

# Visualization
plt.figure(figsize=(12, 8))

# Original Images
plt.subplot(3, 4, 1)
plt.imshow(good_image, cmap="gray")
plt.title("Good Tire (Original)")
plt.axis("off")

plt.subplot(3, 4, 5)
plt.imshow(defective_image, cmap="gray")
plt.title("Defective Tire (Original)")
plt.axis("off")

# LBP Images
plt.subplot(3, 4, 2)
plt.imshow(good_lbp, cmap="gray")
plt.title("LBP - Good Tire")
plt.axis("off")

plt.subplot(3, 4, 6)
plt.imshow(defective_lbp, cmap="gray")
plt.title("LBP - Defective Tire")
plt.axis("off")

# Wavelet Approximation (LL)
plt.subplot(3, 4, 3)
plt.imshow(good_LL, cmap="gray")
plt.title("DWT (LL) - Good Tire")
plt.axis("off")

plt.subplot(3, 4, 7)
plt.imshow(defective_LL, cmap="gray")
plt.title("DWT (LL) - Defective Tire")
plt.axis("off")

# Wavelet Horizontal Details (HL)
plt.subplot(3, 4, 4)
plt.imshow(good_HL, cmap="gray")
plt.title("DWT (HL) - Good Tire")
plt.axis("off")

plt.subplot(3, 4, 8)
plt.imshow(defective_HL, cmap="gray")
plt.title("DWT (HL) - Defective Tire")
plt.axis("off")

# Wavelet Vertical Details (LH)
plt.subplot(3, 4, 9)
plt.imshow(good_LH, cmap="gray")
plt.title("DWT (LH) - Good Tire")
plt.axis("off")

plt.subplot(3, 4, 10)
plt.imshow(defective_LH, cmap="gray")
plt.title("DWT (LH) - Defective Tire")
plt.axis("off")

# Wavelet Diagonal Details (HH)
plt.subplot(3, 4, 11)
plt.imshow(good_HH, cmap="gray")
plt.title("DWT (HH) - Good Tire")
plt.axis("off")

plt.subplot(3, 4, 12)
plt.imshow(defective_HH, cmap="gray")
plt.title("DWT (HH) - Defective Tire")
plt.axis("off")

plt.tight_layout()
plt.show()
