import argparse
import os
import pickle
from torchvision import datasets
from augment.handler import ModelHandler
from augment.utils import Utils
from augment.diffuseMix import DiffuseMix
from PIL import Image

# Define CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

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
    # Parse the arguments
    args = parse_arguments()
    prompts = args.prompts.split(',')  # This will give you a list of prompts

    # Initialize the model
    model_id = "timbrooks/instruct-pix2pix"
    model_initialization = ModelHandler(model_id=model_id, device='cuda')

    # Load CIFAR-10 dataset and convert it to ImageFolder format
    cifar10_dir = args.train_dir  # CIFAR-10 directory (where the batch files are located)
    output_dir = './train/cifar-10'  # Where the organized dataset will go
    # create_image_folder_structure(cifar10_dir, output_dir)  # Convert CIFAR-10

    # Load the original dataset in ImageFolder format
    train_dataset = datasets.ImageFolder(root=args.train_dir)
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}

    # Load fractal images
    fractal_imgs = Utils.load_fractal_images(args.fractal_dir)

    # Create the augmented dataset
    augmented_train_dataset = DiffuseMix(
        original_dataset=train_dataset,
        fractal_imgs=fractal_imgs,
        num_images=1,
        guidance_scale=4,
        idx_to_class=idx_to_class,
        prompts=prompts,
        model_handler=model_initialization
    )

    # Save augmented images
    os.makedirs('augmented_images', exist_ok=True)
    for idx, (image, label) in enumerate(augmented_train_dataset):
        image.save(f'augmented_images/{idx}.png')
        pass

if __name__ == '__main__':
    main()
