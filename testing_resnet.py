import os
import shutil
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, SubsetRandomSampler,Subset
from tqdm import tqdm
import random
from torch.utils.data import Dataset


def write_to_file(text):
    """
    Appends the given text to 'training_results.txt', ensuring each entry is on a new line.
    
    Parameters:
        text (str): The text to be written to the file.
    """
    with open("resnet_training.txt", "a") as file:
        file.write(text + "\n")


# Paths
BASE_DIR = "tiny-imagenet-200"  # Directory for annotations, wnids, words
TRAIN_DIR = "tiny-imagenet-200/train"  # Actual train data directory
ANNOTATIONS_FILE = os.path.join(BASE_DIR, "val", "val_annotations.txt")
WORDS_FILE = os.path.join(BASE_DIR, "words.txt")
WNIDS_FILE = os.path.join(BASE_DIR, "wnids.txt")

# Directories for augmented and blended data
AUGMENTED_DATA_DIR = "result/generated"
BLENDED_DATA_DIR = "result/blended"

# Number of classes to fine-tune
NUM_CLASSES = 20

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Parse Files
def parse_wnids(wnids_file, num_classes):
    with open(wnids_file, "r") as file:
        wnids = [line.strip() for line in file.readlines()]
    return wnids[:num_classes]  # Select the first NUM_CLASSES

def parse_val_annotations(annotations_file):
    annotations = {}
    with open(annotations_file, "r") as file:
        for line in file.readlines():
            parts = line.strip().split("\t")
            image_filename, class_id = parts[0], parts[1]
            annotations[image_filename] = class_id
    return annotations

def parse_words(words_file):
    class_descriptions = {}
    with open(words_file, "r") as file:
        for line in file.readlines():
            wnid, description = line.strip().split("\t", 1)
            class_descriptions[wnid] = description
    return class_descriptions

# Step 2: Prepare Test Dataset (Test images for final evaluation)
import random

def prepare_test_set(train_dir, selected_wnids, num_samples_per_class=75, output_dir="obscured_test_set"):
    """
    Prepares a test set by randomly selecting a specified number of samples per class from the train directory.

    Args:
        train_dir (str): The directory where the original train images are stored.
        selected_wnids (list): A list of selected wnids for the classes to use.
        num_samples_per_class (int): Number of samples to select per class.
        output_dir (str): The directory where the test images should be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create directories for each selected wnid (class)
    for wnid in selected_wnids:
        wnid_dir = os.path.join(output_dir, wnid)
        if not os.path.exists(wnid_dir):
            os.makedirs(wnid_dir)

    # Loop through the selected wnids (classes)
    for wnid in selected_wnids:
        wnid_dir = os.path.join(train_dir, wnid)
        
        if not os.path.exists(wnid_dir):
            print(f"Warning: {wnid_dir} not found in the train set. Skipping this class.")
            continue

        # Get all image files in this class
        all_images = os.listdir(wnid_dir)
        selected_images = random.sample(all_images, num_samples_per_class)  # Select random images

        # Copy the selected images to the new structure
        for image_filename in selected_images:
            src_path = os.path.join(wnid_dir, image_filename)
            dst_path = os.path.join(output_dir, wnid, image_filename)
            shutil.copy(src_path, dst_path)

    print(f"Test set prepared and saved at {output_dir}")

def add_obscuring_transform(image):
    """
    Applies transformations that obscure the features of an image.
    
    Args:
        image (Tensor): The image tensor to which obscuring transformations will be applied.
    
    Returns:
        Tensor: The transformed image tensor.
    """
    # Randomly apply a set of transformations to obscure the image
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Random flip with 50% probability
        transforms.RandomRotation(degrees=15),  # Slight random rotation
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Apply random Gaussian blur
    ])
    return transform(image)


class ObscuredImageDataset(Dataset):
    def __init__(self, dataset, selected_samples):
        """
        A custom dataset class that applies obscuring transformations to images.
        
        Args:
            dataset (Dataset): The dataset to which obscuring transformations will be applied.
            selected_samples (list): A list of (image_path, label) pairs for the selected images.
        """
        self.dataset = dataset
        self.selected_samples = selected_samples

    def __len__(self):
        return len(self.selected_samples)

    def __getitem__(self, idx):
        """
        Get an obscured version of an image and its label.
        
        Args:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            Tuple: (obscured_image, label)
        """
        img, label = self.dataset[self.selected_samples[idx][0]]
        obscured_img = add_obscuring_transform(img)
        return obscured_img, label
    
import os
import shutil
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import random
from PIL import Image

from collections import defaultdict
import random

def collect_samples_from_loader(train_loader, num_samples_per_class=75):
    """
    Collects a specified number of samples per class from the train_loader.
    
    Args:
        train_loader (DataLoader): The DataLoader for the training data.
        num_samples_per_class (int): The number of samples to collect per class.
    
    Returns:
        dict: A dictionary where keys are class indices, and values are lists of image paths and labels.
    """
    class_samples = defaultdict(list)

    # Iterate over the train_loader to collect samples
    for images, labels in tqdm(train_loader, desc="Collecting samples from train_loader"):
        for i in range(len(labels)):
            label = labels[i].item()  # Convert tensor label to int
            if len(class_samples[label]) < num_samples_per_class:
                class_samples[label].append((images[i], label))  # Store image and label pair
                
            # Stop early if we've collected enough samples for each class
            if all(len(class_samples[class_id]) >= num_samples_per_class for class_id in class_samples):
                break
        
        # Check if we have collected enough samples
        if all(len(class_samples[class_id]) >= num_samples_per_class for class_id in class_samples):
            break

    return class_samples

# Function to create a new dataset with obscured images
def create_obscured_dataset_from_loader(class_samples):
    """
    Create a custom dataset of obscured images based on selected samples.

    Args:
        class_samples (dict): A dictionary containing class-wise image samples.

    Returns:
        Dataset: A custom dataset containing the obscured images.
    """
    obscured_samples = []
    
    # Apply obscuring transformations to selected images
    for class_id, samples in class_samples.items():
        for image, label in samples:
            obscured_image = add_obscuring_transform(image)  # Apply transformation
            obscured_samples.append((obscured_image, label))
    
    return obscured_samples


# Creating a custom Dataset for Obscured Images
class ObscuredDataset(Dataset):
    def __init__(self, obscured_samples):
        self.obscured_samples = obscured_samples

    def __len__(self):
        return len(self.obscured_samples)

    def __getitem__(self, idx):
        image, label = self.obscured_samples[idx]
        return image, label




# Step 3: Prepare Dataloaders for Different Datasets (train, augmented, blended)
def create_dataloaders(train_dir, selected_wnids, augmented_dir=None, blended_dir=None, batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(p=0.2),  # Mild flip augmentation with low probability
        transforms.RandomRotation(degrees=10),  # Mild rotation (±10 degrees)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    


    # Prepare the train dataset
    train_dataset = ImageFolder(train_dir, transform=transform)
    print(f"Total samples in train dataset: {len(train_dataset.samples)}")
    print(f"Unique classes in train dataset: {len(set([label for _, label in train_dataset.samples]))}")
    # Map wnids to indices
    wnid_to_class_id = {wnid: idx for idx, wnid in enumerate(selected_wnids)}

    # Filter samples based on selected WNIDs
    filtered_samples = [
        (path, wnid_to_class_id[train_dataset.classes[label]])
        for path, label in train_dataset.samples
        if train_dataset.classes[label] in selected_wnids
    ]
    print(f"Filtered samples size: {len(filtered_samples)}")
    train_dataset.samples = filtered_samples
    train_dataset.targets = [label for _, label in filtered_samples]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Prepare augmented dataset if available
    augmented_loader = None
    if augmented_dir:
        augmented_dataset = ImageFolder(augmented_dir, transform=transform)
        augmented_dataset.samples = [
            (path, wnid_to_class_id[augmented_dataset.classes[label]])
            for path, label in augmented_dataset.samples
            if augmented_dataset.classes[label] in selected_wnids
        ]
        augmented_loader = DataLoader(augmented_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Prepare blended dataset if available
    blended_loader = None
    if blended_dir:
        blended_dataset = ImageFolder(blended_dir, transform=transform)
        blended_dataset.samples = [
            (path, wnid_to_class_id[blended_dataset.classes[label]])
            for path, label in blended_dataset.samples
            if blended_dataset.classes[label] in selected_wnids
        ]
        blended_loader = DataLoader(blended_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, augmented_loader, blended_loader

def fine_tune_resnet(mode, modal_x, test_loader, train_loader, num_classes, num_epochs=3, lr=1e-3):
    # Load pre-trained ResNet-18 model
    # model = None
    # if mode == 'normal':
    #     print("Mode: normal. Initializing ResNet-18.")
    #     model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust output layer for NUM_CLASSES
    # else:
    #     print("Mode: custom. Using provided model.")
    #     model = modal_x
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    #     model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust output layer for NUM_CLASSES
    model = model.to(DEVICE)

    # Debug: Ensure the data loaders are properly loaded
    print(f"Length of train_loader: {len(train_loader)}")
    print(f"Length of test_loader: {len(test_loader)}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Total samples per epoch: {len(train_loader) * train_loader.batch_size}")

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}: Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # Evaluate model after each epoch
        test_accuracy = evaluate_model(model, test_loader)
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        write_to_file(f"Test Accuracy: {test_accuracy:.2f}%")
    return model

def create_mixed_dataloader(original_loader, augmented_loader, ratio=2, batch_size=64, num_workers=4):
    """
    Creates a mixed dataloader with a specified ratio of original to augmented data.
    
    Args:
        original_loader (DataLoader): Dataloader for the original dataset.
        augmented_loader (DataLoader): Dataloader for the augmented dataset.
        ratio (int): Ratio of original to augmented samples.
        batch_size (int): Batch size for the mixed dataloader.
        num_workers (int): Number of workers for the dataloader.
    
    Returns:
        DataLoader: A dataloader containing the mixed dataset.
    """
    original_dataset = original_loader.dataset
    augmented_dataset = augmented_loader.dataset
    
    # Determine number of samples to include
    augmented_size = len(augmented_dataset)
    original_size = ratio * augmented_size
    
    # Randomly sample from the original dataset
    original_indices = random.sample(range(len(original_dataset)), original_size)
    sampled_original_dataset = Subset(original_dataset, original_indices)
    
    # Combine datasets
    combined_dataset = ConcatDataset([sampled_original_dataset, augmented_dataset])
    
    # Create dataloader
    mixed_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return mixed_loader


# Step 5: Evaluate Model
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            # print(f"Length of test_loader: {len(test_loader)}")
            # print(f"Batch size: {test_loader.batch_size}")
            # print(f"Total samples processed per epoch: {len(test_loader) * test_loader.batch_size}")
            # print(f"Number of batches in train_loader: {len(test_loader)}")
            # Print the image names during evaluation
            # for i, path in enumerate(test_loader.dataset.samples):
            #         print(f"Evaluating image name: {os.path.basename(path[0])}")

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Main Script
if __name__ == "__main__":
    # Parse WNIDs and prepare the dataset
    selected_wnids = parse_wnids(WNIDS_FILE, NUM_CLASSES)
    val_annotations = parse_val_annotations(ANNOTATIONS_FILE)
    class_descriptions = parse_words(WORDS_FILE)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 20)  # Adjust output layer for NUM_CLASSES
    model = model.to(DEVICE)


    # Create dataloaders for normal, augmented, and blended data
    print("Creating dataloaders...")
    train_loader, augmented_loader, blended_loader = create_dataloaders(
        TRAIN_DIR, selected_wnids, augmented_dir=AUGMENTED_DATA_DIR, blended_dir=BLENDED_DATA_DIR, batch_size=100
    )
    print("Collecting samples from train_loader...")
    class_samples = collect_samples_from_loader(train_loader, num_samples_per_class=75)
    print("Creating obscured dataset...")
    obscured_samples = create_obscured_dataset_from_loader(class_samples)
    mixed_aug_loader=create_mixed_dataloader(train_loader, augmented_loader, ratio=2)
    mixed_blend_loader=create_mixed_dataloader(train_loader, blended_loader, ratio=2)
    # Create a DataLoader for the obscured dataset
    obscured_dataset = ObscuredDataset(obscured_samples)
    test_loader = DataLoader(obscured_dataset, batch_size=64, shuffle=True, num_workers=4)
    for i in range(50):
      write_to_file("Training on normal data")
      model_y= fine_tune_resnet("normal",model,test_loader,augmented_loader, num_classes=NUM_CLASSES, num_epochs=3)
      final_accuracy = evaluate_model(model, test_loader)
      print(f"Test Accuracy (Augmented data): {final_accuracy:.2f}%")
      if augmented_loader:
        print("Training on augmented data...")
        write_to_file("Training on Augmented data")
        model_z = fine_tune_resnet("test",model_y,test_loader,mixed_aug_loader, num_classes=NUM_CLASSES, num_epochs=3)
        final_accuracy = evaluate_model(model, test_loader)
        print(f"Test Accuracy (Augmented data): {final_accuracy:.2f}%")

      # Fine-tune model on blended data
      write_to_file("Training on blended data")
      if blended_loader:
        print("Training on blended data...")
        model_y = fine_tune_resnet("test",model_y,test_loader,mixed_blend_loader, num_classes=NUM_CLASSES, num_epochs=3)
        final_accuracy = evaluate_model(model, test_loader)

    