import os
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


split_dataset("../../../datasets/EuroSAT_RGB")