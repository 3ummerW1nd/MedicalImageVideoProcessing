# 数字图像处理项目一

### 1.技术：tensorflow、keras、unet模型......

### 2.目录结构：

​	logs文件夹存放训练过程的输出

​	model_data文件夹存放需要的h5模型数据

​	nets文件存放的是有关unet模型实现的文件

​	new_image存放的是处理后每一帧图片

​	original_image存放的是处理前的每一帧图片

​	utils存放的是计算权重的工具

​	video存放的是原始的视频以及处理后输出的视频

​	VOCdevkit/VOC2007中存放的是训练模型用到的数据，其中：

​		JPGEImages存放训练数据的原始图像

​		SegmentaClass存放利用labelme标记产生的权重图像

​		运行voc_2_unet.py之后将会在ImageSets/Segmentation中生成训练所需要的txt文件

​	divide.py用来分隔图像

​	imgs_to_video.py用来把new_image存放的所有图片按顺序拼成新的视频

​	train.py用来利用数据集训练

​	unet.py存放了训练中所需要利用unet模型进行的操作

​	video_2_imgs.py用来把原始视频分成一个个帧

### 3.操作说明：

##### （1）训练：

​		a. 将利用labelme标记好的数据复制到指定目录（参考上面的目录结构）

​		b. 运行VOCdevkit/voc_2_unet.py 以生成利用voc训练必要的txt文件

​		c. 运行train.py

​		d. 在logs文件夹中查看训练的结果

##### （2）处理：

​		a. 将原视频放在video目录下之后，运行video_2_imgs.py以将原视频按帧分为图片

​		b. 运行divide.py，在new_image文件夹生成分割后的图片（分割所用的默认权值文件是我之前利用这个视频的前32张图片训练的，如果想换成自己训练的版本，可以将文件重命名为my_voc.h5放在model_data文件夹中替换掉原有权值文件）

​		c. 运行imgs_2_video将分割好的图片拼接成视频，前往video文件夹中查看new_video.mp4即为最终结果

​		