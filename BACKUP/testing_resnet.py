import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from torch import nn, optim

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset class with hardcoded labels
class ImageDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        """
        Args:
            image_dir (str): Directory with all the images.
            label (str): Label for all images in this directory ('airplane' or 'automobile').
            transform (callable, optional): Transform to apply to images.
        """
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)  # List of all images in the directory
        self.label = 0 if label == "airplane" else 1  # Hardcoded label
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])  # Image file path
        image = Image.open(img_name).convert("RGB")  # Ensure 3-channel images
        if self.transform:
            image = self.transform(image)
        return image, self.label

# Define image preprocessing
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing to ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])

# Create DataLoader
def get_dataloaders(image_dirs, labels, batch_size=32, train_split=0.8):
    datasets = []
    for image_dir, label in zip(image_dirs, labels):
        datasets.append(ImageDataset(image_dir=image_dir, label=label, transform=get_transform()))
    
    # Combine all datasets into one
    combined_dataset = torch.utils.data.ConcatDataset(datasets)
    
    # Split into training and validation
    train_size = int(len(combined_dataset) * train_split)
    val_size = len(combined_dataset) - train_size
    train_set, val_set = random_split(combined_dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# Define ResNet model
def initialize_model(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify last layer
    return model.to(device)

# Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=10):
    train_loader, val_loader = dataloaders
    for epoch in range(num_epochs):
        # Training phase
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
        
        # Validation phase
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

# Evaluate model accuracy
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    print(dataloader)
    with torch.no_grad():
        for images, labels in dataloader[1]:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)  # Get predicted class
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

# Main function to train and compare models
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ResNet models on original and augmented datasets.")
    parser.add_argument("--original_airplane", type=str, required=True, help="Path to original airplane images.")
    parser.add_argument("--original_automobile", type=str, required=True, help="Path to original automobile images.")
    parser.add_argument("--augmented_airplane", type=str, required=True, help="Path to augmented airplane images.")
    parser.add_argument("--augmented_automobile", type=str, required=True, help="Path to augmented automobile images.")
    parser.add_argument("--testData_airplane",type=str,required=True,help="Path to test data")
    parser.add_argument("--testData_automobile",type=str,required=True,help="Path to test data")
    args = parser.parse_args()

    # Load datasets for original only
    original_dataloaders = get_dataloaders(
        image_dirs=[args.original_airplane, args.original_automobile],
        labels=["airplane", "automobile"]
    )
    
    # Load datasets for combined original + augmented
    combined_dataloaders = get_dataloaders(
        image_dirs=[args.original_airplane, args.original_automobile, args.augmented_airplane, args.augmented_automobile],
        labels=["airplane", "automobile", "airplane", "automobile"]
    )
    
    test_dataloaders = get_dataloaders(
        image_dirs=[args.testData_airplane, args.testData_automobile],
        labels=["airplane", "automobile"]
    )


    # Initialize the ResNet model
    num_classes = 2  # Airplane and automobile
    model = initialize_model(num_classes)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # A) Train on the original dataset
    print("Training on Original Dataset")
    train_model(model, original_dataloaders, criterion, optimizer)
    
    # B) Evaluate on the original dataset after training on the original data
    print("\nEvaluating on Original Dataset after training on Original Dataset")
    original_accuracy = evaluate_model(model, test_dataloaders)  # Validation set
    print(f"Accuracy on Original Dataset: {original_accuracy:.2f}%")
    
    # C) Train on the combined original + augmented dataset
    print("\nTraining on Combined Original and Augmented Dataset")
    train_model(model, combined_dataloaders, criterion, optimizer)
    
    # D) Evaluate on the original dataset after training on both original and augmented datasets
    print("\nEvaluating on Original Dataset after training on Combined Dataset")
    final_accuracy = evaluate_model(model, test_dataloaders)  # Validation set (original dataset)
    print(f"Final Accuracy on Original Dataset (after combined training): {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
