import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch import nn, optim
import csv

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
        transforms.Resize((224, 224)),  # Resizing to fixed size
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

# Define the Base CNN model
class BaseCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # First convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Second convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Third convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.fc1 = nn.Linear(64 * 28 * 28, 512)  # First fully connected layer
        self.fc2 = nn.Linear(512, num_classes)  # Second fully connected layer for classification

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))  # First convolution + ReLU + pooling
        x = self.pool(nn.ReLU()(self.conv2(x)))  # Second convolution + ReLU + pooling
        x = self.pool(nn.ReLU()(self.conv3(x)))  # Third convolution + ReLU + pooling
        x = x.view(-1, 64 * 28 * 28)  # Flatten the tensor before feeding to fully connected layers
        x = nn.ReLU()(self.fc1(x))  # Fully connected layer with ReLU activation
        x = self.fc2(x)  # Final output layer
        return x

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

# Evaluate function to check accuracy on original dataset
def evaluate_model(model, dataloaders):
    model.eval()
    val_loader = dataloaders[1]
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"Accuracy on Original Dataset: {acc:.2f}%")
    return "{acc:.2f}"
# Main function to train and compare models
def main():
    parser = argparse.ArgumentParser(description="Train a base CNN model on original and augmented datasets.")
    parser.add_argument("--original_airplane", type=str, required=True, help="Path to original airplane images.")
    parser.add_argument("--original_automobile", type=str, required=True, help="Path to original automobile images.")
    parser.add_argument("--augmented_airplane", type=str, required=True, help="Path to augmented airplane images.")
    parser.add_argument("--augmented_automobile", type=str, required=True, help="Path to augmented automobile images.")
    parser.add_argument("--testData_airplane",type=str,required=True,help="Path to test data")
    parser.add_argument("--testData_automobile",type=str,required=True,help="Path to test data")
    args = parser.parse_args()

    # Load datasets
    original_dataloaders = get_dataloaders(
        image_dirs=[args.original_airplane, args.original_automobile],
        labels=["airplane", "automobile"]
    )
    augmented_dataloaders = get_dataloaders(
        image_dirs=[args.augmented_airplane, args.augmented_automobile],
        labels=["airplane", "automobile"]
    )
    
    test_dataloaders = get_dataloaders(
        image_dirs=[args.testData_airplane, args.testData_automobile],
        labels=["airplane", "automobile"]
    )

    # Initialize the Base CNN model
    num_classes = 2  # Airplane and automobile
    model_original = BaseCNN(num_classes=num_classes).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_original = optim.Adam(model_original.parameters(), lr=0.001)
    with open('training_results_base.csv', 'w', newline='') as csvfile:
        fieldnames = ['Training Run', 'Eval #', 'Original Dataset Accuracy', 'Augmented+Original Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 5 training runs
        for run in range(1, 6):
            print(f"\nTraining Run {run}")
            # Step A: Train on the original dataset
            train_model(model_original, original_dataloaders, criterion, optimizer_original)

    
            original_acc = evaluate_model(model_original, test_dataloaders)

            writer.writerow({'Training Run': run,  'Original Dataset Accuracy': original_acc})
        
        for run in range(1, 11):
            print(f"\nTraining Run {run}")
            # Step A: Train on the original dataset
            combined_dataloaders = get_dataloaders(
                    image_dirs=[args.original_airplane, args.original_automobile, args.augmented_airplane, args.augmented_automobile],
                    labels=["airplane", "automobile", "airplane", "automobile"]
                )
            train_model(model_original, combined_dataloaders, criterion, optimizer_original)

            combined_acc = evaluate_model(model_original, test_dataloaders)

            writer.writerow({'Training Run': run, 'Augmented+Original Accuracy': combined_acc})

if __name__ == "__main__":
    main()

