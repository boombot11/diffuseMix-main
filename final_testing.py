import argparse
import os
import torch
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from PIL import Image, ImageFilter
from torch import nn, optim
import numpy as np

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset class with hardcoded labels
class ImageDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.label = 0 if label == "airplane" else 1
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label

# Define image preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Apply augmentations to generate test data
def generate_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(30),  # Random rotation between -30 and 30 degrees
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),  # Random blur
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random noise
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Create DataLoader with ratio
def get_dataloaders_with_ratio(
    image_dirs, labels, augmented_dirs=None, batch_size=32, train_split=0.8
):
    datasets = []
    for image_dir, label in zip(image_dirs, labels):
        datasets.append(ImageDataset(image_dir=image_dir, label=label, transform=get_transform()))

    combined_dataset = torch.utils.data.ConcatDataset(datasets)

    if augmented_dirs:
        augmented_datasets = []
        for augmented_dir, label in zip(augmented_dirs, labels):
            augmented_datasets.append(ImageDataset(image_dir=augmented_dir, label=label, transform=get_transform()))
        
        augmented_combined = torch.utils.data.ConcatDataset(augmented_datasets)
        normal_indices = list(range(len(combined_dataset)))
        augmented_indices = list(range(len(augmented_combined)))

        np.random.shuffle(normal_indices)
        np.random.shuffle(augmented_indices)

        total_augmented = len(augmented_combined)
        normal_size = 2 * total_augmented
        normal_subset = Subset(combined_dataset, normal_indices[:normal_size])
        augmented_subset = Subset(augmented_combined, augmented_indices)

        combined_dataset = torch.utils.data.ConcatDataset([normal_subset, augmented_subset])

    train_size = int(len(combined_dataset) * train_split)
    val_size = len(combined_dataset) - train_size
    train_set, val_set = random_split(combined_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader



# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    train_loader, val_loader = dataloaders
    for epoch in range(10):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        train_acc = 100 * correct / total

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.2f}%")
        write_to_file(f"Epoch {epoch+1}/{num_epochs}: Train Loss={train_loss/len(train_loader):.4f}, Train Acc={train_acc:.2f}%, Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.2f}%")
    return model

# Evaluate function
def evaluate_model(model, test_loader):  # Change parameter name to reflect test_loader
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Accuracy on Dataset: {acc:.2f}%")
    write_to_file(f"Test Accuracy: {acc:.2f}%")

    

# Generate test data using the training data (with augmentations)
def generate_test_data_from_train(train_loader):
    augmented_test_dataset = []
    for images, labels in train_loader:
        # Apply test augmentations
        for img, label in zip(images, labels):
            # Convert the Tensor image back to a PIL Image for applying augmentations
            img_pil = transforms.ToPILImage()(img)  # Convert the Tensor to PIL Image
            
            augmented_img = generate_test_transform()(img_pil)  # Apply augmentations
            augmented_test_dataset.append((augmented_img, label))
    return augmented_test_dataset

def write_to_file(text):
    """
    Appends the given text to 'training_results.txt', ensuring each entry is on a new line.
    
    Parameters:
        text (str): The text to be written to the file.
    """
    with open("cifar_normal.txt", "a") as file:
        file.write(text + "\n")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a base CNN model on original and augmented datasets.")
    parser.add_argument("--original_airplane", type=str, required=True, help="Path to original airplane images.")
    parser.add_argument("--original_automobile", type=str, required=True, help="Path to original automobile images.")
    parser.add_argument("--augmented_airplane_a1", type=str, required=True, help="Path to augmented airplane images A1.")
    parser.add_argument("--augmented_automobile_a1", type=str, required=True, help="Path to augmented automobile images A1.")
    parser.add_argument("--augmented_airplane_a2", type=str, required=True, help="Path to augmented airplane images A2.")
    parser.add_argument("--augmented_automobile_a2", type=str, required=True, help="Path to augmented automobile images A2.")
    parser.add_argument("--testData_airplane", type=str, required=True, help="Path to test airplane images.")
    parser.add_argument("--testData_automobile", type=str, required=True, help="Path to test automobile images.")
    args = parser.parse_args()

    # Load original dataloaders
    original_dataloaders = get_dataloaders_with_ratio(
        image_dirs=[args.original_airplane, args.original_automobile],
        labels=["airplane", "automobile"]
    )

    # Load augmented dataloaders for A1
    a1_dataloaders = get_dataloaders_with_ratio(
        image_dirs=[args.original_airplane, args.original_automobile],
        labels=["airplane", "automobile"],
        augmented_dirs=[args.augmented_airplane_a1, args.augmented_automobile_a1]
    )

    # Load augmented dataloaders for A2
    a2_dataloaders = get_dataloaders_with_ratio(
        image_dirs=[args.original_airplane, args.original_automobile],
        labels=["airplane", "automobile"],
        augmented_dirs=[args.augmented_airplane_a2, args.augmented_automobile_a2]
    )

    # Test dataloaders - generated from training data
    augmented_test_data = generate_test_data_from_train(a1_dataloaders[0])  # Test from A1 data
    augmented_test_loader = DataLoader(augmented_test_data, batch_size=32, shuffle=False)

    # Initialize the Base CNN model
    model = models.resnet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for i in range(50):
     print("Training on normal data")
     write_to_file("NORMAL TRAINING")
     train_model(model,original_dataloaders,criterion,optimizer)
     evaluate_model(model,augmented_test_loader)
     # Train and evaluate on A1
     print("Training on Dataset with A1 Augmented Data")
     write_to_file("A1 TRAINING")
    #  result= train_model(model,original_dataloaders,criterion,optimizer)
     train_model(model, a1_dataloaders, criterion, optimizer)
     print("\nEvaluating on Test Dataset after training on A1 Augmented Data")
     evaluate_model(model, augmented_test_loader)

     # Train and evaluate on A2
     model = models.resnet18().to(device) # Reinitialize model
     optimizer = optim.Adam(model.parameters(), lr=0.001)
     print("\nTraining on Dataset with A2 Augmented Data")
     write_to_file("A2 TRAINING")
    #  res= train_model(model,original_dataloaders,criterion,optimizer)
     train_model(model, a2_dataloaders, criterion, optimizer)
     print("\nEvaluating on Test Dataset after training on A2 Augmented Data")
     evaluate_model(model, augmented_test_loader)

if __name__ == "__main__":
    main()
