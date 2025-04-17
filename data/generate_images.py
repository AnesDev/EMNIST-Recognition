import random
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from augmentations import apply_augmentations

IMG_SIZE = (280, 280)
FONT_DIR = "fonts"
DIGITS = "0123456789"
UPPERCASE_CHARACTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE_CHARACTERS = "abcdefghijklmnopqrstuvwxyz"

def load_fonts(font_dir):
    font_files = [os.path.join(font_dir, file) for file in os.listdir(font_dir)]
    return font_files


def generate_image(character, font, img_size=IMG_SIZE):
    # Create a blank black image in grayscale
    img = Image.new('L', img_size, 0)
    draw = ImageDraw.Draw(img)

    font_size = random.randint(80, 150)
    font = ImageFont.truetype(font, font_size)

    # Calculating bounding box of the text
    bbox = draw.textbbox((0, 0), character, font=font)
    text_width = bbox[2] - bbox[0]  # Width of character
    text_height = bbox[3] - bbox[1]  # Height of character

    # Draw the character in the center
    position = ((img_size[0] - text_width) // 2, (img_size[1] - text_height) // 2)
    draw.text(position, character, font=font, fill=255)  # White character in grayscale

    augmented_images = apply_augmentations(img) # Apply augmenttaions

    return augmented_images




def save_image(image, char, font_name, save_dir="dataset"):
    if char in UPPERCASE_CHARACTERS:
        folder_name = f"upper_{char}"
    elif char in LOWERCASE_CHARACTERS:
        folder_name = f"lower_{char}"
    else:
        folder_name = f"digit_{char}"

    # Full path
    char_dir = os.path.join(save_dir, folder_name)
    os.makedirs(char_dir, exist_ok=True)

    # Save each augmented version 
    for idx, img in enumerate(image):
        file_name = f"{folder_name}_{font_name}_{idx}.png"
        file_path = os.path.join(char_dir, file_name)
        img.save(file_path)
        print(f"Saved image to {file_path}")




fonts = load_fonts(FONT_DIR)

for font in fonts:
    for char_set in [UPPERCASE_CHARACTERS, LOWERCASE_CHARACTERS, DIGITS]:
        for char in char_set:
            generated_images = generate_image(char, font)
            save_image(generated_images, char, os.path.basename(font))