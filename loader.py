import os
import random
from shutil import copyfile

# Paths to Tiny ImageNet dataset
tiny_imagenet_dir = './tiny-imagenet-200/'  # Base directory of Tiny ImageNet
train_images_dir = os.path.join(tiny_imagenet_dir, 'train')  # Training images path
output_dir = './FINAL/Tiny_imagenet_200_Data/'  # Output directory for filtered images

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to load 75 images per class
def load_75_images_per_class():
    # Get all class directories in the training folder
    class_dirs = [d for d in os.listdir(train_images_dir) if os.path.isdir(os.path.join(train_images_dir, d))]

    for class_dir in class_dirs:
        class_path = os.path.join(train_images_dir, class_dir, 'images')  # Path to class-specific images
        if not os.path.exists(class_path):
            print(f"Images folder not found for class: {class_dir}")
            continue

        # Get all image files in the class folder
        image_files = [f for f in os.listdir(class_path) if f.endswith('.JPEG')]

        # Randomly select 75 images
        selected_images = random.sample(image_files, min(len(image_files), 75))

        # Create class folder in the output directory
        class_output_dir = os.path.join(output_dir, class_dir)
        os.makedirs(class_output_dir, exist_ok=True)

        # Copy selected images to the output directory
        for image_file in selected_images:
            src_path = os.path.join(class_path, image_file)
            dest_path = os.path.join(class_output_dir, image_file)
            copyfile(src_path, dest_path)

    print("75 images per class have been saved to the output directory.")

# Run the function to load and save images
load_75_images_per_class()
