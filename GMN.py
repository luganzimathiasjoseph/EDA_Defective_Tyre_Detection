import zipfile
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# Paths 
extract_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of defective and good condition tyres"

good_tires_dir = os.path.join(extract_dir, "good")
defective_tires_dir = os.path.join(extract_dir, "defective")


# Store extracted features
good_features = []
defective_features = []

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
        
        # Feature extraction
        noise = np.std(gray_image)  # Noise estimation
        brightness = np.mean(gray_image)  # Brightness estimation
        intensity = np.percentile(gray_image, 90)  # Light intensity estimation
        
        feature_vector = [noise, brightness, intensity]
        
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

# Combine and create labels
X = np.vstack((good_features, defective_features))
y = np.hstack((np.zeros(len(good_features)), np.ones(len(defective_features))))  # 0: Good, 1: Defective

# Fit Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)
predictions = gmm.predict(X)

# Classification report
accuracy = np.mean(predictions == y)
print(f"GMM Classification Accuracy: {accuracy * 100:.2f}%")

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm', alpha=0.7)
plt.title("GMM Classification of Tire Condition")
plt.xlabel("Noise Level")
plt.ylabel("Brightness Level")
plt.colorbar(label='Cluster')
plt.show()

print("Gaussian Mixture Model classification completed!")
