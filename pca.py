import zipfile
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paths 
extract_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of defective and good condition tyres"


good_tires_dir = os.path.join(extract_dir, "good")
defective_tires_dir = os.path.join(extract_dir, "defective")


# Store extracted features
good_glcm_features = []
defective_glcm_features = []
good_hog_features = []
defective_hog_features = []

# Processing images
def process_images(folder_path, label):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg"))]
    print(f"Processing {len(image_files)} {label} tire images...")
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        image = cv2.imread(img_path)
        
        if image is None:
            continue
        
        image = cv2.resize(image, (256, 256))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # GLCM Graphs
        glcm = graycomatrix(gray_image, distances=[1], angles=[0], symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        correlation = graycoprops(glcm, "correlation")[0, 0]
        
        # HOG Graphs
        fd, _ = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        
        if label == "good":
            good_glcm_features.append([contrast, correlation])
            good_hog_features.append(np.mean(fd))
        else:
            defective_glcm_features.append([contrast, correlation])
            defective_hog_features.append(np.mean(fd))

# Process images
process_images(good_tires_dir, "good")
process_images(defective_tires_dir, "defective")

# Convert lists to NumPy arrays
good_glcm_features = np.array(good_glcm_features)
defective_glcm_features = np.array(defective_glcm_features)
good_hog_features = np.array(good_hog_features)
defective_hog_features = np.array(defective_hog_features)

# Combine the features into a single dataset
good_features = np.hstack((good_glcm_features, good_hog_features.reshape(-1, 1)))
defective_features = np.hstack((defective_glcm_features, defective_hog_features.reshape(-1, 1)))

# Labels
good_labels = np.zeros(good_features.shape[0])  # Label 0 for good tires
defective_labels = np.ones(defective_features.shape[0])  # Label 1 for defective tires

# Combine the data and labels
X = np.vstack((good_features, defective_features))
y = np.hstack((good_labels, defective_labels))

# Perform PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_pca = pca.fit_transform(X)

# Plot the PCA results
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color='green', label='Good Tires', alpha=0.6)
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color='red', label='Defective Tires', alpha=0.6)
plt.title('PCA of Tire Features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

print("PCA visualization completed!")

# Proceed with the rest of your code for classification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Classification report
print(classification_report(y_test, y_pred))
