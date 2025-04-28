import os
from torch.utils.data import Dataset
from PIL import Image
import random
from augment.utils import Utils
import numpy as np
from scipy.ndimage import gaussian_filter

class DiffuseMix(Dataset):
    def __init__(self, original_dataset, num_images, guidance_scale, fractal_imgs, idx_to_class, prompts, model_handler):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.combine_counter = 0
        self.fractal_imgs = fractal_imgs
        self.prompts = prompts
        self.model_handler = model_handler
        self.num_augmented_images_per_image = num_images
        self.guidance_scale = guidance_scale
        self.utils = Utils()
        self.augmented_images = self.generate_augmented_images()

    @staticmethod
    def generate_saliency_mask(original_img: Image.Image, factor=0.45, sigma=0.5, threshold=0.5):
        """
        Generate a binary mask where important (salient) regions are 1, rest are 0.
        """
        image_np = np.asarray(original_img.convert('L')) / 255.0
        saliency_map = np.power(image_np, factor)
        saliency_map = gaussian_filter(saliency_map, sigma=sigma)

        saliency_map = saliency_map - np.min(saliency_map)
        saliency_map = saliency_map / np.max(saliency_map)

        binary_mask = (saliency_map > threshold).astype(np.uint8)  # salient = 1, non-salient = 0
        return binary_mask

    @staticmethod
    def overlay_images(base_image, overlay_image):
        """
        Simple weighted blend of two images.
        """
        base_np = np.asarray(base_image).astype(float) / 255.0
        overlay_np = np.asarray(overlay_image).astype(float) / 255.0
        composite_np = (0.6 * base_np + 0.4 * overlay_np)
        composite_np = (composite_np * 255).astype(np.uint8)
        return Image.fromarray(composite_np)

    @staticmethod
    def smart_blend_with_mask(base_img: Image.Image, overlay_img: Image.Image, mask: np.ndarray):
        """
        Blend only the non-salient regions using the given binary mask.
        """
        base_np = np.asarray(base_img).astype(np.float32) / 255.0
        overlay_np = np.asarray(overlay_img).astype(np.float32) / 255.0

        # Expand mask to 3 channels
        mask_3ch = np.stack([mask] * 3, axis=-1)

        # Where mask == 1 (salient), keep base_img
        # Where mask == 0 (non-salient), blend base and overlay
        blended_np = base_np * mask_3ch + (0.6 * base_np + 0.4 * overlay_np) * (1 - mask_3ch)

        blended_np = np.clip(blended_np, 0, 1)
        blended_np = (blended_np * 255).astype(np.uint8)
        return Image.fromarray(blended_np)

    def generate_augmented_images(self):
        augmented_data = []

        base_directory = './result'
        generated_dir = os.path.join(base_directory, 'generated')
        fractal_dir = os.path.join(base_directory, 'fractal')
        concatenated_dir = os.path.join(base_directory, 'concatenated')
        blended_dir = os.path.join(base_directory, 'blended')
        saliency_dir = os.path.join(base_directory, 'saliency')
        composite_dir = os.path.join(base_directory, 'composite')

        os.makedirs(generated_dir, exist_ok=True)
        os.makedirs(fractal_dir, exist_ok=True)
        os.makedirs(concatenated_dir, exist_ok=True)
        os.makedirs(blended_dir, exist_ok=True)
        os.makedirs(saliency_dir, exist_ok=True)
        os.makedirs(composite_dir, exist_ok=True)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):
            label = self.idx_to_class[label_idx]  # Label as folder name

            # Load and resize original image
            original_img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_filename = os.path.basename(img_path)
            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in
                          ['generated', 'fractal', 'concatenated', 'blended', 'saliency', 'composite']}

            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            for prompt in self.prompts:
                # Generate augmented images
                augmented_images = self.model_handler.generate_images(
                    prompt, img_path, self.num_augmented_images_per_image, self.guidance_scale
                )

                for i, gen_img in enumerate(augmented_images):
                    gen_img = gen_img.resize((256, 256))
                    generated_img_filename = f"{img_filename}_generated_{prompt}_{i}.jpg"
                    gen_img.save(os.path.join(label_dirs['generated'], generated_img_filename))

                    if not self.utils.is_black_image(gen_img):
                        # Step 1: Generate binary saliency mask
                        saliency_mask = self.generate_saliency_mask(original_img)
                        saliency_mask_img = Image.fromarray((saliency_mask * 255).astype(np.uint8)).convert('RGB')
                        saliency_map_filename = f"{img_filename}_saliency_{prompt}_{i}.jpg"
                        saliency_mask_img.save(os.path.join(label_dirs['saliency'], saliency_map_filename))

                        # Step 2: Overlay original and generated
                        composite_img = self.overlay_images(gen_img, original_img)
                        composite_img_filename = f"{img_filename}_composite_{prompt}_{i}.jpg"
                        composite_img.save(os.path.join(label_dirs['composite'], composite_img_filename))

                        # Step 3: Pick random fractal
                        random_fractal_img = random.choice(self.fractal_imgs).resize((256, 256))
                        fractal_img_filename = f"{img_filename}_fractal_{prompt}_{i}.jpg"
                        random_fractal_img.save(os.path.join(label_dirs['fractal'], fractal_img_filename))

                        # Step 4: Smart blend based on saliency
                        smart_blended_img = self.smart_blend_with_mask(composite_img, random_fractal_img, saliency_mask)
                        smart_blended_img_filename = f"{img_filename}_blended_{prompt}_{i}.jpg"
                        smart_blended_img.save(os.path.join(label_dirs['blended'], smart_blended_img_filename))

                        augmented_data.append((smart_blended_img, label))

        return augmented_data

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        image, label = self.augmented_images[idx]
        return image, label
