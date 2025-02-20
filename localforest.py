import zipfile
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Paths 
zip_path = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Digital images of defective and good condition tyres.zip"
extract_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of defective and good condition tyres"

good_tires_dir = os.path.join(extract_dir, "good")
defective_tires_dir = os.path.join(extract_dir, "defective")

# Ensure extraction directory exists
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction completed!")

# Store extracted features
good_features = []
defective_features = []

# Processing images
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
        
        # GLCM Features
        glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        correlation = graycoprops(glcm, "correlation")[0, 0]
        
        # HOG Features
        fd, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog_mean = np.mean(fd)  # Convert HOG descriptor into a single feature
        
        feature_vector = [contrast, correlation, hog_mean]
        
        if label == "good":
            good_features.append(feature_vector)
        else:
            defective_features.append(feature_vector)

# Process images
process_images(good_tires_dir, "good")
process_images(defective_tires_dir, "defective")

# Convert lists to NumPy arrays
good_features = np.array(good_features)
defective_features = np.array(defective_features)

# Combine features into one dataset for outlier detection
all_features = np.vstack([good_features, defective_features])

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(all_features)

# Apply Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_iso = iso_forest.fit_predict(scaled_features)

# Apply LOF for outlier detection
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
outliers_lof = lof.fit_predict(scaled_features)

# Visualize the outliers for both methods
plt.figure(figsize=(16, 12))

# Isolation Forest
plt.subplot(2, 2, 1)
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=outliers_iso, cmap='coolwarm')
plt.title('Isolation Forest for Outlier Detection')
plt.xlabel('GLCM Contrast')
plt.ylabel('GLCM Correlation')

# LOF
plt.subplot(2, 2, 2)
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=outliers_lof, cmap='coolwarm')
plt.title('Local Outlier Factor (LOF) for Outlier Detection')
plt.xlabel('GLCM Contrast')
plt.ylabel('GLCM Correlation')

# Plot GLCM Contrast Distribution
plt.subplot(2, 2, 3)
plt.hist(good_features[:, 0], bins=30, color='green', alpha=0.7, label='Good Tires')
plt.hist(defective_features[:, 0], bins=30, color='red', alpha=0.7, label='Defective Tires')
plt.title("GLCM Contrast Distribution")
plt.xlabel("Contrast")
plt.ylabel("Frequency")
plt.legend()

# Plot GLCM Correlation Distribution
plt.subplot(2, 2, 4)
plt.hist(good_features[:, 1], bins=30, color='green', alpha=0.7, label='Good Tires')
plt.hist(defective_features[:, 1], bins=30, color='red', alpha=0.7, label='Defective Tires')
plt.title("GLCM Correlation Distribution")
plt.xlabel("Correlation")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

print("Feature extraction, outlier detection, and visualization completed!")
