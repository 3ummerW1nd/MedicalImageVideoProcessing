import cv2 as cv
import os
from PIL import Image
import numpy as np

images_path = "new_image"
video_path = "video/new_video.mp4"
fps = 10
images_list = os.listdir(images_path)
image = Image.open(os.path.join(images_path, images_list[0]))
image_size = image.size
fourcc = cv.VideoWriter_fourcc(*'XVID') # 意为四字符代码
video_writer = cv.VideoWriter(video_path, fourcc, fps, image_size)
for i in images_list:
    image_name = os.path.join(images_path + '/' + i)
    frame = cv.imdecode(np.fromfile(image_name, dtype=np.uint8), -1)
    video_writer.write(frame)
video_writer.release()