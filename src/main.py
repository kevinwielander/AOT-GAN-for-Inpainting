import os
import torch
import cv2
import numpy as np
from glob import glob
from torchvision.transforms import ToTensor
from utils.option import args
import importlib

class ImageInpainter:
    def __init__(self, model_name, pretrained_model_path, image_dir, output_dir):
        self.model_name = model_name
        self.pretrained_model_path = pretrained_model_path
        self.image_dir = image_dir
        self.output_dir = os.path.join(output_dir, 'inpainting_results')
        os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

    def load_model(self):
        net = importlib.import_module("model." + self.model_name)
        self.model = net.InpaintGenerator(args)
        self.model.load_state_dict(torch.load(self.pretrained_model_path, map_location=self.device))
        self.model.eval()

    def postprocess(self, image):
        image = torch.clamp(image, -1.0, 1.0)
        image = (image + 1) / 2.0 * 255.0
        image = image.permute(1, 2, 0)
        image = image.cpu().numpy().astype(np.uint8)
        return image

    def inpaint_image(self, image_path, mask_path):
        orig_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize and normalize
        img_tensor = ToTensor()(orig_img).unsqueeze(0) * 2 - 1
        mask_tensor = ToTensor()(mask).unsqueeze(0)

        # Perform inpainting
        with torch.no_grad():
            masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
            pred_tensor = self.model(masked_tensor, mask_tensor)
            comp_tensor = pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor)

            # Postprocess and save
            comp_np = self.postprocess(comp_tensor[0])
            filename = os.path.basename(image_path).split('.')[0]
            output_path = os.path.join(self.output_dir, f'{filename}_inpainted.png')
            cv2.imwrite(output_path, comp_np)
            print(f'Inpainted image saved to {output_path}')

    def process_directory(self):
        img_list = [f for f in glob(os.path.join(self.image_dir, '*.png')) if '_mask' not in f]
        for img_path in img_list:
            mask_path = img_path.replace('.png', '_mask.png')
            if os.path.exists(mask_path):
                self.inpaint_image(img_path, mask_path)
            else:
                print(f'Mask not found for {img_path}')

# Example usage
if __name__ == "__main__":
    inpainter = ImageInpainter('aotgan', '../experiments/G0000000.pt', 'input', 'output/results')
    inpainter.process_directory()
