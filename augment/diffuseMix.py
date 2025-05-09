import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
from augment.utils import Utils

class DiffuseMix(Dataset):
    def __init__(self, original_dataset, num_images, guidance_scale, fractal_imgs, idx_to_class, prompts, model_handler):
        self.original_dataset = original_dataset
        self.idx_to_class = idx_to_class
        self.fractal_imgs = fractal_imgs
        self.prompts = prompts
        self.model_handler = model_handler
        self.num_augmented_images_per_image = num_images
        self.guidance_scale = guidance_scale
        self.utils = Utils()
        self.augmented_images = self.generate_augmented_images()

    @staticmethod
    def generate_soft_saliency_map(original_img: Image.Image, factor=0.45, sigma=1.5):
        """
        Generate a soft saliency map as float values between 0-1.
        """
        gray_np = np.asarray(original_img.convert('L')) / 255.0
        saliency_map = np.power(gray_np, factor)
        saliency_map = gaussian_filter(saliency_map, sigma=sigma)

        saliency_map -= saliency_map.min()
        saliency_map /= (saliency_map.max() + 1e-8)
        return saliency_map

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
    def smart_blend_with_soft_mask(base_img: Image.Image, overlay_img: Image.Image, mask: np.ndarray):
        """
        Blend images using a soft (float-valued) saliency mask.
        """
        base_np = np.asarray(base_img).astype(np.float32) / 255.0
        overlay_np = np.asarray(overlay_img).astype(np.float32) / 255.0
        mask_3ch = np.stack([mask] * 3, axis=-1)

        blended_np = base_np * mask_3ch + (0.6 * base_np + 0.4 * overlay_np) * (1 - mask_3ch)
        blended_np = np.clip(blended_np, 0, 1)
        return Image.fromarray((blended_np * 255).astype(np.uint8))

    def is_destructive_gan_output(self, gan_img: Image.Image, original_img: Image.Image, threshold=0.85):
        """
        Check if GAN output has completely overridden original image content.
        Returns True if > threshold of pixels are too different from the original.
        """
        gan_np = np.asarray(gan_img.resize((256, 256))).astype(np.float32) / 255.0
        orig_np = np.asarray(original_img.resize((256, 256))).astype(np.float32) / 255.0

        diff = np.abs(gan_np - orig_np)
        high_diff_ratio = (diff > 0.5).mean()

        return high_diff_ratio > threshold

    def generate_augmented_images(self):
        augmented_data = []

        base_directory = './result'
        subdirs = ['generated', 'fractal', 'concatenated', 'blended', 'saliency', 'composite']
        for subdir in subdirs:
            os.makedirs(os.path.join(base_directory, subdir), exist_ok=True)

        for idx, (img_path, label_idx) in enumerate(self.original_dataset.samples):
            label = self.idx_to_class[label_idx]
            original_img = Image.open(img_path).convert('RGB').resize((256, 256))
            img_filename = os.path.basename(img_path)

            label_dirs = {dtype: os.path.join(base_directory, dtype, str(label)) for dtype in subdirs}
            for dir_path in label_dirs.values():
                os.makedirs(dir_path, exist_ok=True)

            for prompt in self.prompts:
                augmented_images = self.model_handler.generate_images(
                    prompt, img_path, self.num_augmented_images_per_image, self.guidance_scale
                )

                for i, gen_img in enumerate(augmented_images):
                    gen_img = gen_img.resize((256, 256))
                    gen_filename = f"{img_filename}_generated_{prompt}_{i}.jpg"
                    gen_img.save(os.path.join(label_dirs['generated'], gen_filename))

                    if self.utils.is_black_image(gen_img):
                        continue  # skip bad GAN output

                    # GAN sanity check — skip images that overwrite everything
                    if self.is_destructive_gan_output(gen_img, original_img):
                        continue

                    # Step 1: Get soft saliency map
                    saliency_map = self.generate_soft_saliency_map(original_img)
                    saliency_img = Image.fromarray((saliency_map * 255).astype(np.uint8)).convert('RGB')
                    saliency_filename = f"{img_filename}_saliency_{prompt}_{i}.jpg"
                    saliency_img.save(os.path.join(label_dirs['saliency'], saliency_filename))

                    # Step 2: Overlay original and generated
                    composite_img = self.overlay_images(gen_img, original_img)
                    composite_filename = f"{img_filename}_composite_{prompt}_{i}.jpg"
                    composite_img.save(os.path.join(label_dirs['composite'], composite_filename))

                    # Step 3: Get a random fractal
                    random_fractal_img = random.choice(self.fractal_imgs).resize((256, 256))
                    fractal_filename = f"{img_filename}_fractal_{prompt}_{i}.jpg"
                    random_fractal_img.save(os.path.join(label_dirs['fractal'], fractal_filename))

                    # Step 4: Smart blend with soft saliency map
                    blended_img = self.smart_blend_with_soft_mask(composite_img, random_fractal_img, saliency_map)
                    blended_filename = f"{img_filename}_blended_{prompt}_{i}.jpg"
                    blended_img.save(os.path.join(label_dirs['blended'], blended_filename))

                    augmented_data.append((blended_img, label))

        return augmented_data

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        return self.augmented_images[idx]
