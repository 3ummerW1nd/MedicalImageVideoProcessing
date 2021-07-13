import time
import os
import cv2
import numpy as np
from PIL import Image

from unet import Unet

if __name__ == "__main__":

    unet = Unet()
    images_list = os.listdir("original_image")
    for i in images_list:
        img = os.path.join("original_image/" + i)
        image = Image.open(img)
        r_image = unet.detect_image(image)
        r_image.save("new_image/" + i)
