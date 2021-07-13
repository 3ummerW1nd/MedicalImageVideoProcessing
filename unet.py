import colorsys
import copy
import numpy as np
from PIL import Image
from nets.unet import Unet as unet


class Unet(object):
    _defaults = {
        "model_path": 'model_data/my_voc.h5',
        "model_image_size": (512, 512, 3),
        "num_classes": 2,
        "blend": True,
    }
# 初始化
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.generate()
# 载入模型
    def generate(self):
        self.model = unet(self.model_image_size, self.num_classes)

        self.model.load_weights(self.model_path)
        print('{} model loaded.'.format(self.model_path))

        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128),
                           (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
                           (192, 0, 128),
                           (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
                           (0, 64, 128), (128, 64, 12)]
        else:
            hsv_tuples = [(x / len(self.class_names), 1., 1.)
                          for x in range(len(self.class_names))]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                    self.colors))

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, nw, nh


    def detect_image(self, image):
        image = image.convert('RGB')
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # 补充灰条以使得图片大小合适
        img, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        img = np.asarray([np.array(img) / 255])
        # 传入网络
        pr = self.model.predict(img)[0]
        # 取出像素点的种类
        pr = pr.argmax(axis=-1).reshape([self.model_image_size[0], self.model_image_size[1]])
        # 截掉之前填充好的灰条
        pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
             int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]

        # 创建新图，根据类别填色
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')

        # 转换成Image
        image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h), Image.NEAREST)

        # 混合图片
        if self.blend:
            image = Image.blend(old_img, image, 0.7)

        return image

