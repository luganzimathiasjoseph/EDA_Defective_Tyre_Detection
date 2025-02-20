import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.measure import shannon_entropy
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths
zip_path = r"C:\Users\HP\Downloads\Compressed\Digital images of defective and good condition tyres.zip"
extract_dir = r"C:\Users\HP\OneDrive\Desktop\tyre quality assessmen using Gans and computer vision\Digital images of defective and good condition tyres"

good_tires_dir = os.path.join(extract_dir, "good")
defective_tires_dir = os.path.join(extract_dir, "defective")

# Extraction directory
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction completed!")

# Store extracted features
good_features = []
defective_features = []

def extract_features(image):
    image = cv2.resize(image, (256, 256))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GLCM Features
    glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    
    # HOG Features
    fd, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    hog_feature = np.mean(fd)
    
    # LBP Features
    lbp = local_binary_pattern(gray_image, P=8, R=1, method="uniform")
    lbp_mean = np.mean(lbp)
    
    # Shannon Entropy
    entropy = shannon_entropy(gray_image)
    
    # Edge Detection
    edges = cv2.Canny(gray_image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    
    return [contrast, correlation, hog_feature, lbp_mean, entropy, contour_count]

def process_images(folder_path, label):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
    print(f"Processing {len(image_files)} {label} tire images...")
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        features = extract_features(image)
        
        if label == "good":
            good_features.append(features)
        else:
            defective_features.append(features)

# Process images
process_images(good_tires_dir, "good")
process_images(defective_tires_dir, "defective")

# Convert lists to NumPy arrays
good_features = np.array(good_features)
defective_features = np.array(defective_features)

# Prepare dataset
X = np.vstack((good_features, defective_features))
y = np.hstack((np.zeros(len(good_features)), np.ones(len(defective_features))))  # 0 for good, 1 for defective

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))

print("Feature extraction, visualization, and classification completed!")
