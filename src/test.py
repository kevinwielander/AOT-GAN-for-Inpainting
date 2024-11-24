import importlib
import os
from glob import glob
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from utils.option import args


def postprocess(image):
    image = torch.clamp(image, -1.0, 1.0)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return Image.fromarray(image)


def recursive_glob(root_dir, pattern):
    return [str(path) for path in Path(root_dir).rglob(pattern)]


def main_worker(args, use_gpu=True):
    # Model and version
    net = importlib.import_module("model." + args.model)
    model = net.InpaintGenerator(args).cuda()
    model.load_state_dict(torch.load(args.pre_train, map_location="cuda"))
    model.eval()

    # prepare dataset
    image_paths = []
    for ext in ["*.jpg", "*.png", "*.JPEG"]:
        image_paths.extend(recursive_glob(os.path.join(args.dir_image, args.data_train), ext))
    image_paths.sort()

    mask_paths = sorted(glob(os.path.join(args.dir_mask, args.mask_type, "*.png")))
    if not mask_paths:
        print(f"No masks found in {os.path.join(args.dir_mask, args.mask_type)}")
        print("Using default quarter mask")
        h = w = args.image_size
        default_mask = np.zeros((h, w), dtype=np.uint8)
        default_mask[:h // 2, :w // 2] = 255

        default_mask_path = os.path.join(args.outputs, "default_mask.png")
        Image.fromarray(default_mask).save(default_mask_path)
        mask_paths = [default_mask_path]

    os.makedirs(args.outputs, exist_ok=True)

    if not image_paths:
        raise ValueError(f"No images found in {os.path.join(args.dir_image, args.data_train)}")

    # iteration through datasets
    for ipath in image_paths:
        # Get random mask for each image
        mpath = np.random.choice(mask_paths)

        image = ToTensor()(Image.open(ipath).convert("RGB"))
        image = (image * 2.0 - 1.0).unsqueeze(0)
        mask = ToTensor()(Image.open(mpath).convert("L"))
        mask = mask.unsqueeze(0)
        image, mask = image.cuda(), mask.cuda()
        image_masked = image * (1 - mask.float()) + mask

        with torch.no_grad():
            pred_img = model(image_masked, mask)

        comp_imgs = (1 - mask) * image + mask * pred_img
        image_name = os.path.basename(ipath).split(".")[0]
        postprocess(image_masked[0]).save(os.path.join(args.outputs, f"{image_name}_masked.png"))
        postprocess(pred_img[0]).save(os.path.join(args.outputs, f"{image_name}_pred.png"))
        postprocess(comp_imgs[0]).save(os.path.join(args.outputs, f"{image_name}_comp.png"))
        print(f"saving to {os.path.join(args.outputs, image_name)}")


if __name__ == "__main__":
    main_worker(args)
