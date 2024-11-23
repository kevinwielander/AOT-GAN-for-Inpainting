import numpy as np
from PIL import Image

image = np.zeros((64, 64, 3), dtype=np.uint8)
image[0:32, 0:32] = [255, 255, 255]
img = Image.fromarray(image)
img.save('mask.png')
img.show()
