import os
from glob import glob

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class InpaintingData(Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.w = self.h = args.image_size
        self.mask_type = args.mask_type

        # image and mask
        self.image_path = []
        for ext in ["*.jpg", "*.png", "*.JPEG"]:
            self.image_path.extend(self.recursive_glob(os.path.join(args.dir_image, args.data_train), ext))

        if not self.image_path:
            raise ValueError(f"No images found in {os.path.join(args.dir_image, args.data_train)}")

        self.mask_path = glob(os.path.join(args.dir_mask, args.mask_type, "*.png"))

        # augmentation
        self.img_trans = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor(),
            ]
        )
        self.mask_trans = transforms.Compose(
            [
                transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((0, 45), interpolation=transforms.InterpolationMode.NEAREST),
            ]
        )

    def recursive_glob(self, root_dir, pattern):
        return [str(path) for path in Path(root_dir).rglob(pattern)]

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        # load image
        image = Image.open(self.image_path[index]).convert("RGB")
        filename = os.path.basename(self.image_path[index])

        if self.mask_type == "pconv":
            index = np.random.randint(0, len(self.mask_path))
            mask = Image.open(self.mask_path[index])
            mask = mask.convert("L")
        else:
            mask = np.zeros((self.h, self.w)).astype(np.uint8)
            mask[:self.h // 2, :self.w // 2] = 1
            mask = Image.fromarray(mask).convert("L")

        # augment
        image = self.img_trans(image) * 2.0 - 1.0
        mask = F.to_tensor(self.mask_trans(mask)) * 255

        return image, mask, filename

if __name__ == "__main__":
    from types import SimpleNamespace

    args = SimpleNamespace(
        dir_image="../../dataset",
        data_train="ilsvrc2012_64x64/train",
        dir_mask="../../dataset",
        mask_type="mask",
        image_size=64
    )

    data = InpaintingData(args)
    print(len(data), len(data.mask_path))
    img, mask, filename = data[0]
    print(img.size(), mask.size(), filename)
