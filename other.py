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
extract_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of defective and good condition tyres"

good_tires_dir = os.path.join(extract_dir, "good")
defective_tires_dir = os.path.join(extract_dir, "defective")


# Store extracted features
good_glcm_features = []
defective_glcm_features = []
good_hog_features = []
defective_hog_features = []

# Store segmentation results for good and defective tires
watershed_results_good = []
watershed_results_defective = []
grabcut_results_good = []
grabcut_results_defective = []

# Watershed segmentation function
def watershed_segmentation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    image[markers == -1] = [255, 0, 0]
    return image, markers  # Return segmented image and markers

# GrabCut segmentation function
def grabcut_segmentation(image):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    result = image * mask2[:, :, np.newaxis]
    return result, mask2  # Return segmented image and mask

# Sample image processing function with segmentation
def process_texture_segmentation(image_path, label):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (200, 200))
    
    # Watershed segmentation
    watershed_img, watershed_markers = watershed_segmentation(image.copy())
    
    # GrabCut segmentation
    grabcut_img, grabcut_mask = grabcut_segmentation(image.copy())
    
    # Store the results based on the label (good or defective)
    if label == "good":
        watershed_results_good.append(watershed_markers)  # Store watershed markers for good tires
        grabcut_results_good.append(grabcut_mask)  # Store grabcut mask for good tires
    else:
        watershed_results_defective.append(watershed_markers)  # Store watershed markers for defective tires
        grabcut_results_defective.append(grabcut_mask)  # Store grabcut mask for defective tires

# Processing images
def process_images(folder_path, label):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg"))]
    print(f"Processing {len(image_files)} {label} tire images...")
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        process_texture_segmentation(img_path, label)

# Process images
process_images(good_tires_dir, "good")
process_images(defective_tires_dir, "defective")

# Plotting results for watershed and grabcut (separated for good and defective tires)
def plot_segmentation_results():
    # Number of regions (max value of markers/mask) for good tires
    num_regions_watershed_good = [np.max(watershed) for watershed in watershed_results_good]
    num_regions_grabcut_good = [np.max(grabcut) for grabcut in grabcut_results_good]
    
    # Number of regions (max value of markers/mask) for defective tires
    num_regions_watershed_defective = [np.max(watershed) for watershed in watershed_results_defective]
    num_regions_grabcut_defective = [np.max(grabcut) for grabcut in grabcut_results_defective]

    # Plot Watershed Region Counts for Good Tires
    plt.figure(figsize=(8, 6))
    plt.hist(num_regions_watershed_good, bins=30, color='blue', alpha=0.7)
    plt.title("Watershed Segmentation (Good Tires): Number of Regions")
    plt.xlabel("Number of Regions")
    plt.ylabel("Frequency")
    plt.show()

    # Plot Watershed Region Counts for Defective Tires
    plt.figure(figsize=(8, 6))
    plt.hist(num_regions_watershed_defective, bins=30, color='orange', alpha=0.7)
    plt.title("Watershed Segmentation (Defective Tires): Number of Regions")
    plt.xlabel("Number of Regions")
    plt.ylabel("Frequency")
    plt.show()

    # Plot GrabCut Region Counts for Good Tires
    plt.figure(figsize=(8, 6))
    plt.hist(num_regions_grabcut_good, bins=30, color='green', alpha=0.7)
    plt.title("GrabCut Segmentation (Good Tires): Number of Regions")
    plt.xlabel("Number of Regions")
    plt.ylabel("Frequency")
    plt.show()

    # Plot GrabCut Region Counts for Defective Tires
    plt.figure(figsize=(8, 6))
    plt.hist(num_regions_grabcut_defective, bins=30, color='red', alpha=0.7)
    plt.title("GrabCut Segmentation (Defective Tires): Number of Regions")
    plt.xlabel("Number of Regions")
    plt.ylabel("Frequency")
    plt.show()

# Call the function to plot the results
plot_segmentation_results()

# Feature extraction and visualization completed
print("Feature extraction and visualization completed!")
