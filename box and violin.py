import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.cluster import KMeans

# Paths
extract_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of defective and good condition tyres"

good_tires_dir = os.path.join(extract_dir, "good")
defective_tires_dir = os.path.join(extract_dir, "defective")

# Store extracted features
features = []
labels = []

def process_images(folder_path, label):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    print(f"Processing {len(image_files)} {label} tire images...")
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        image = cv2.resize(image, (256, 256))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Extract GLCM Features
        glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        correlation = graycoprops(glcm, "correlation")[0, 0]
        
        # Extract HOG Features
        fd, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_mean = np.mean(fd)
        
        # Store features
        features.append([contrast, correlation, hog_mean, label])

# Process images from both categories
process_images(good_tires_dir, "good")
process_images(defective_tires_dir, "defective")

# Convert features to a DataFrame
df = pd.DataFrame(features, columns=["GLCM Contrast", "GLCM Correlation", "HOG Mean", "Label"])

# Box Plot for Good and Defective Tires
plt.figure(figsize=(10, 6))
sns.boxplot(x="Label", y="GLCM Contrast", data=df, palette="coolwarm")
plt.title("Box Plot of GLCM Contrast for Good and Defective Tires")
plt.xlabel("Tire Condition")
plt.ylabel("GLCM Contrast")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="Label", y="GLCM Correlation", data=df, palette="coolwarm")
plt.title("Box Plot of GLCM Correlation for Good and Defective Tires")
plt.xlabel("Tire Condition")
plt.ylabel("GLCM Correlation")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="Label", y="HOG Mean", data=df, palette="coolwarm")
plt.title("Box Plot of HOG Mean for Good and Defective Tires")
plt.xlabel("Tire Condition")
plt.ylabel("HOG Mean")
plt.show()

# Violin Plot for Good and Defective Tires
plt.figure(figsize=(10, 6))
sns.violinplot(x="Label", y="GLCM Contrast", data=df, palette="coolwarm")
plt.title("Violin Plot of GLCM Contrast for Good and Defective Tires")
plt.xlabel("Tire Condition")
plt.ylabel("GLCM Contrast")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x="Label", y="GLCM Correlation", data=df, palette="coolwarm")
plt.title("Violin Plot of GLCM Correlation for Good and Defective Tires")
plt.xlabel("Tire Condition")
plt.ylabel("GLCM Correlation")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x="Label", y="HOG Mean", data=df, palette="coolwarm")
plt.title("Violin Plot of HOG Mean for Good and Defective Tires")
plt.xlabel("Tire Condition")
plt.ylabel("HOG Mean")
plt.show()

print("Feature extraction, clustering, and visualization completed!")
