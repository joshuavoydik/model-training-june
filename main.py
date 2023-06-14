import torch
from torchvision import transforms, datasets, models
from PIL import Image
from sklearn.cluster import DBSCAN
import cv2
import numpy as np
import os

# Load the pre-trained model
model = models.resnet18(pretrained=True)
model = model.eval()  # Set the model to evaluation mode

# Define a transformation function that will be applied on images
transform = transforms.Compose([
    transforms.Resize(256), 
    transforms.CenterCrop(224), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')  # Convert grayscale images to RGB
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img

# Directory containing your images
image_dir = "./caltech-101/101_ObjectCategories"

# List to store feature vectors
feature_list = []

# List to store corresponding image paths
image_path_list = []

# Process all images in the directory
for category in os.listdir(image_dir):
    category_path = os.path.join(image_dir, category)
    
    # Skip if not a directory
    if not os.path.isdir(category_path):
        continue
    
    print(f"Processing category: {category}")
    
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        
        # Skip if not an image
        if not image_path.endswith(('.png', '.jpg', '.jpeg')):
            print(f"Skipping non-image file: {image_path}")
            continue

        print(f"Processing image: {image_path}")
        
        # Preprocess the image
        img = preprocess_image(image_path)

        # Extract features from the image
        with torch.no_grad():  # No need to compute gradients
            features = model(img)

        # Store the features and corresponding image path
        feature_list.append(features.cpu().numpy().ravel())
        image_path_list.append(image_path)

# Check if any features were extracted
if not feature_list:
    print("No features were extracted.")
    exit(1)

# Convert feature tensors to numpy arrays
feature_array = np.array(feature_list)

# Perform clustering
clustering = DBSCAN(eps=0.3, min_samples=2).fit(feature_array)

# Get cluster labels for each image
labels = clustering.labels_

# Iterate over each cluster (exclude outliers)
for cluster_id in set(labels):
    if cluster_id == -1:
        continue

    print(f"Cluster {cluster_id}:")
    
    # Print image paths in the same cluster
    for image_path, label in zip(image_path_list, labels):
        if label == cluster_id:
            print(f"    {image_path}")

