import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
extract_dir = r"C:\Users\MATT\Desktop\CS2\Course Work\Machine Learning\Images\Digital images of defective and good condition tyres"
good_tires_dir = os.path.join(extract_dir, "good")
defective_tires_dir = os.path.join(extract_dir, "defective")

# Store edge detection results
good_cracks = []
defective_cracks = []

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
        
        # Canny Edge Detection
        edges = cv2.Canny(gray_image, 100, 200)
        
        # Calculate the number of edges (cracks) detected
        num_edges = np.sum(edges == 255)
        
        if label == "good":
            good_cracks.append(num_edges)
        else:
            defective_cracks.append(num_edges)

# Process images from both categories
process_images(good_tires_dir, "good")
process_images(defective_tires_dir, "defective")

# Bar Graph of Crack Identification (Edge Detection)
plt.figure(figsize=(8, 6))

# Plot for Good Tires
plt.bar(["Good Tires"] * len(good_cracks), good_cracks, color='blue', alpha=0.7, label='Good Tires')

# Plot for Defective Tires
plt.bar(["Defective Tires"] * len(defective_cracks), defective_cracks, color='red', alpha=0.7, label='Defective Tires')

plt.xlabel("Tire Condition")
plt.ylabel("Number of Cracks Detected (Edges)")
plt.title("Crack Identification using Edge Detection (Canny)")
plt.legend()
plt.show()

print("Edge detection for crack identification and bar graph visualization completed!")
