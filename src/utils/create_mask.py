import numpy as np
from PIL import Image

image = np.zeros((256, 256, 3), dtype=np.uint8)
image[0:128, 0:128] = [255, 255, 255]
img = Image.fromarray(image)
img.save('mask_256x256.png')
img.show()
