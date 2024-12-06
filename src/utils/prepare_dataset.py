import os
from PIL import Image
import random
import shutil
from pathlib import Path


def split_dataset(root_folder, train_ratio=0.8, seed=2024):
    random.seed(seed)

    root_path = Path(root_folder)
    new_root = root_path.parent / f"{root_path.name}_transformed"
    new_root.mkdir(exist_ok=True)

    # Create train and test folders
    train_folder = new_root / "train"
    test_folder = new_root / "test"
    train_folder.mkdir(exist_ok=True)
    test_folder.mkdir(exist_ok=True)

    # Iterate through category folders
    for category_folder in root_path.iterdir():
        if category_folder.is_dir():
            # Create corresponding category folders in train and test
            train_category = train_folder / category_folder.name
            test_category = test_folder / category_folder.name
            train_category.mkdir(exist_ok=True)
            test_category.mkdir(exist_ok=True)

            types = ['*.jpg', '*.png', '*.jpeg']
            image_files = []
            for t in types:
                image_files.extend(list(category_folder.glob(t)))

            if len(image_files) < 10:
                print(f"Warning: Not enough files in {category_folder.name} for proper splitting.")
                continue

            random.shuffle(image_files)

            split_index = int(len(image_files) * train_ratio)

            for i, image_file in enumerate(image_files):
                if i < split_index:
                    destination = train_category / image_file.name
                else:
                    destination = test_category / image_file.name
                shutil.copy2(image_file, destination)

    print(f"Dataset split complete. New dataset location: {new_root}")

def preprocess_images(dataset_path, target_height, target_width):
    output_dir = f"{dataset_path}_{target_height}x{target_width}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    # Check if the file is an image
                    if img.format not in ["JPEG", "JPG", "PNG", "BMP", "GIF", "TIFF"]:
                        continue

                    # Center crop the image
                    width, height = img.size
                    left = (width - target_width) // 2
                    top = (height - target_height) // 2
                    right = left + target_width
                    bottom = top + target_height

                    cropped_img = img.crop((left, top, right, bottom))

                    # Preserve folder structure in the output directory
                    relative_path = os.path.relpath(root, dataset_path)
                    output_subdir = os.path.join(output_dir, relative_path)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)

                    # Save the cropped image
                    output_file_path = os.path.join(output_subdir, file)
                    if os.path.exists(output_file_path):
                        print(f"Overwriting existing file: {output_file_path}")
                    cropped_img.save(output_file_path)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")


preprocess_images("../../../datasets/ILSVRC2012_train", 256, 256)
# split_dataset("../../../datasets/EuroSAT_RGB")