import zipfile
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report 

# Paths 
zip_path = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Digital images of defective and good condition tyres.zip"
extract_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of dsefective and good condition tyre"

good_tires_dir = os.path.join(extract_dir, "good")
defective_tires_dir = os.path.join(extract_dir, "defective")

# Extraction directory
os.makedirs(extract_dir, exist_ok=True)

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction completed!")

# Store extracted features
good_noise = []
defective_noise = []
good_brightness = []
defective_brightness = []
good_intensity = []
defective_intensity = []

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
        
        # Noise estimation
        noise = np.std(gray_image)
        
        # Brightness estimation
        brightness = np.mean(gray_image)
        
        # Light intensity estimation
        intensity = np.percentile(gray_image, 90)
        
        if label == "good":
            good_noise.append(noise)
            good_brightness.append(brightness)
            good_intensity.append(intensity)
        else:
            defective_noise.append(noise)
            defective_brightness.append(brightness)
            defective_intensity.append(intensity)

# Process images
process_images(good_tires_dir, "good")
process_images(defective_tires_dir, "defective")

# Convert lists to NumPy arrays
good_noise = np.array(good_noise)
defective_noise = np.array(defective_noise)
good_brightness = np.array(good_brightness)
defective_brightness = np.array(defective_brightness)
good_intensity = np.array(good_intensity)
defective_intensity = np.array(defective_intensity)

# Noise Distribution
plt.figure(figsize=(6, 5))
plt.hist(good_noise, bins=30, color='green', alpha=0.7, label='Good Tires')
plt.hist(defective_noise, bins=30, color='red', alpha=0.7, label='Defective Tires')
plt.title("Noise Distribution")
plt.xlabel("Noise Level")
plt.ylabel("Frequency")
plt.legend()
plt.show()  # Show the first plot and wait for it to be closed

# Brightness Distribution
plt.figure(figsize=(6, 5))
plt.hist(good_brightness, bins=30, color='blue', alpha=0.7, label='Good Tires')
plt.hist(defective_brightness, bins=30, color='orange', alpha=0.7, label='Defective Tires')
plt.title("Brightness Distribution")
plt.xlabel("Brightness Level")
plt.ylabel("Frequency")
plt.legend()
plt.show()  # Show the second plot and wait for it to be closed

# Light Intensity Distribution
plt.figure(figsize=(6, 5))
plt.hist(good_intensity, bins=30, color='purple', alpha=0.7, label='Good Tires')
plt.hist(defective_intensity, bins=30, color='brown', alpha=0.7, label='Defective Tires')
plt.title("Light Intensity Distribution")
plt.xlabel("Intensity Level")
plt.ylabel("Frequency")
plt.legend()
plt.show()  # Show the third plot and wait for it to be closed

print("Feature extraction and visualization completed!")
