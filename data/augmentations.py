import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter


def apply_rotation(image):
    angle = random.uniform(-30, 30)
    return image.rotate(angle)

def apply_blur(image):
    if random.random() < 0.5:
        return image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.5)))
    return image

def apply_shear(image):
    max_shear = 0.2
    shear = random.uniform(-max_shear, max_shear)
    return image.transform(image.size, Image.AFFINE, (1, shear, 0, shear, 1, 0))

def apply_zoom(image):
    zoom_factor = random.uniform(0.8, 1.2)
    width, height = image.size
    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)
    return image.resize((new_width, new_height))

def apply_noise(image):
    np_image = np.array(image)
    noise = np.random.normal(0, 0.1, np_image.shape)
    np_image = np.clip(np_image + noise * 50, 0, 255).astype(np.uint8)
    return Image.fromarray(np_image)


def apply_augmentations(image):
    augmented_images = []
    
    for _ in range(7):  # Loop 7 times to apply 7 augmentations
        new_image = image.copy()
        
        # Randomly choose and apply an augmentation
        augmentation_choice = random.choice([apply_rotation, apply_blur])  # You can add more augmentations here
        new_image = augmentation_choice(new_image)
        
        augmented_images.append(new_image)

    return augmented_images