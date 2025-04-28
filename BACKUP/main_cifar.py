import os
import sys
import random
from collections import defaultdict
from PIL import Image

import torch
from torchvision import datasets

# Add the path to your local augment module
sys.path.append('/content/DiffuseMix/')

from augment.handler import ModelHandler
from augment.utils import Utils
from augment.diffuseMix import DiffuseMix

# Config
TRAIN_DIR = '/content/train/cifar-10'       # Existing ImageFolder-structured CIFAR-10
FRACTAL_DIR = '/content/deviantart/'        # Fractal image directory
OUTPUT_DIR = '/content/result'              # Where augmented images will be saved
IMAGES_PER_CLASS = 50                       # Number of images to augment per class
PROMPTS = ['winter', 'autumm']               # Modify if needed

# Load CIFAR-10 dataset
full_dataset = datasets.ImageFolder(root=TRAIN_DIR)

# Group image paths by class
class_to_images = defaultdict(list)
for img_path, label in full_dataset.samples:
    class_to_images[label].append(img_path)

# Select N images per class
selected_images, selected_labels = [], []
for label, imgs in class_to_images.items():
    chosen = random.sample(imgs, min(IMAGES_PER_CLASS, len(imgs)))
    selected_images.extend(chosen)
    selected_labels.extend([label] * len(chosen))

# Define custom dataset (with .samples support for DiffuseMix)
class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.samples = list(zip(image_paths, labels))  # ✅ Make it compatible with DiffuseMix
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.image_paths[index]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.image_paths)

# Wrap selected images in dataset
train_dataset = CustomImageDataset(selected_images, selected_labels)
idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

# Load fractals
fractal_imgs = Utils.load_fractal_images(FRACTAL_DIR)

# Device & Model Initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

# Monkey patch: patch ModelHandler to use float32 on CPU inside handler.py or update here if needed
model_handler = ModelHandler(model_id="timbrooks/instruct-pix2pix", device=device)

# Create augmented dataset
augmented_dataset = DiffuseMix(
    original_dataset=train_dataset,
    fractal_imgs=fractal_imgs,
    num_images=1,
    guidance_scale=4,
    idx_to_class=idx_to_class,
    prompts=PROMPTS,
    model_handler=model_handler
)

# Save results
for idx, (image, label) in enumerate(augmented_dataset):
    class_name = idx_to_class[label]
    save_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(save_path, exist_ok=True)
    image.save(os.path.join(save_path, f'{idx}.png'))

print("✅ Augmentation complete. Check:", OUTPUT_DIR)
