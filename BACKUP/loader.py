import os
import pickle
from torchvision import datasets
import numpy as np
from PIL import Image

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Only include 'airplane' and 'automobile' (indices 0 and 1)
target_classes = [0, 1]  # 'airplane' and 'automobile'

# Paths
cifar10_dir = './train/cifar-10-batches-py'  # Path to the CIFAR-10 dataset
output_dir = './Final/cifar-10'  # Where the organized dataset will go

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to load CIFAR-10 batches
def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='bytes')

# Convert the CIFAR-10 dataset into the ImageFolder format
def create_image_folder_structure():
    # Load all 5 batches of CIFAR-10
    for batch_num in range(1, 6):
        batch_file = os.path.join(cifar10_dir, f'data_batch_{batch_num}')
        batch_data = unpickle(batch_file)
        data = batch_data[b'data']  # Shape: (10000, 3072)
        labels = batch_data[b'labels']  # Shape: (10000,)

        # Reshape the images and save only for 'airplane' (0) and 'automobile' (1)
        for i, label in enumerate(labels):
            if label in target_classes:  # Only process airplane and automobile images
                image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape to (32, 32, 3)
                image = Image.fromarray(image_data)

                # Create the class folder if it doesn't exist
                class_folder = os.path.join(output_dir, class_names[label])
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                # Save the image in the corresponding class folder
                image.save(os.path.join(class_folder, f'{i + batch_num * 10000}.png'))

# Run the function to create the dataset structure
create_image_folder_structure()

print("Dataset conversion complete!")
