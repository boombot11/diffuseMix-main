import argparse
import os
import pickle
import torch
from torchvision import datasets
from augment.handler import ModelHandler
from augment.utils import Utils
from augment.diffuseMix import DiffuseMix
from PIL import Image
from collections import defaultdict

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Custom wrapper to mimic Subset but preserve .samples
class SubsetWithSamples(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.samples = [dataset.samples[i] for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate an augmented dataset from original images and fractal patterns.")
    parser.add_argument('--train_dir', type=str, required=True, help='Path to the directory containing the original training images.')
    parser.add_argument('--fractal_dir', type=str, required=True, help='Path to the directory containing the fractal images.')
    parser.add_argument('--prompts', type=str, required=True, help='Comma-separated list of prompts for image generation.')
    return parser.parse_args()

# Function to load CIFAR-10 batches
def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='bytes')

# Convert the CIFAR-10 dataset into the ImageFolder format
def create_image_folder_structure(cifar10_dir, output_dir, max_images_per_class=500):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize counters for each class
    class_counters = {class_name: 0 for class_name in class_names}

    # Load all 5 batches of CIFAR-10
    for batch_num in range(1, 6):
        batch_file = os.path.join(cifar10_dir, f'data_batch_{batch_num}')
        batch_data = unpickle(batch_file)
        data = batch_data[b'data']  # Shape: (10000, 3072)
        labels = batch_data[b'labels']  # Shape: (10000,)

        # Reshape the images and save them into class-specific folders
        for i, label in enumerate(labels):
            if class_counters[class_names[label]] >= max_images_per_class:
                continue  # Skip this image if the class has already reached the limit

            image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)  # Reshape to (32, 32, 3)
            image = Image.fromarray(image_data)

            # Create the class folder if it doesn't exist
            class_folder = os.path.join(output_dir, class_names[label])
            if not os.path.exists(class_folder):
                os.makedirs(class_folder)

            # Save the image in the corresponding class folder
            image.save(os.path.join(class_folder, f'{class_counters[class_names[label]] + 1}.png'))

            # Update the counter for the current class
            class_counters[class_names[label]] += 1

            # Stop processing this batch if all classes have 5,000 images
            if all(count >= max_images_per_class for count in class_counters.values()):
                return

def main():
    args = parse_arguments()
    prompts = args.prompts.split(',')

    # Initialize model
    model_id = "timbrooks/instruct-pix2pix"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Using device: {device}")
    model_initialization = ModelHandler(model_id=model_id, device=device)

    # Load and prepare dataset
    cifar10_dir = args.train_dir
    output_dir = './train/cifar-10'
    # create_image_folder_structure(cifar10_dir, output_dir)

    train_dataset = datasets.ImageFolder(root=output_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(args.fractal_dir)

    # Organize dataset by class
    class_to_images = defaultdict(list)
    for img_path, label in train_dataset.samples:
        class_to_images[label].append((img_path, label))

    os.makedirs('augmented_images', exist_ok=True)
    total_augmented = 0

    for class_label, samples in class_to_images.items():
        print(f"Generating for class: {idx_to_class[class_label]}")
        selected_samples = samples[:10]  # Select 10 images per class
        subset_indices = [train_dataset.samples.index(s) for s in selected_samples]

        # Use custom wrapper to keep .samples attribute
        subset_dataset = SubsetWithSamples(train_dataset, subset_indices)

        # Generate 5 augmentations per image × 10 images = 50 total
        augmenter = DiffuseMix(
            original_dataset=subset_dataset,
            num_images=5,
            guidance_scale=4,
            fractal_imgs=fractal_imgs,
            idx_to_class=idx_to_class,
            prompts=prompts,
            model_handler=model_initialization
        )

        for i, (image, label) in enumerate(augmenter):
            class_name = idx_to_class[label]
            image.save(f'augmented_images/{class_name}_{i}.png')
            total_augmented += 1

    print(f"Total augmented images saved: {total_augmented}")

if __name__ == '__main__':
    main()
