#!/bin/bash/env python
#-*- coding: utf-8
__author__ = 'ZhangJin'

'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def func(x, y, sigma=1):
    return 100*(1/(2*np.pi*sigma))*np.exp(-((x-2)**2+(y-2)**2)/(2.0*sigma**2))


suanzi1 = np.fromfunction(func, (5, 5), sigma=5)

# Laplace扩展算子
suanzi2 = np.array([[1, 1, 1],
                    [1, -8, 1],
                    [1, 1, 1]])


def imconv(image_array, suanzi):
    image = image_array.copy()
    dim1, dim2 = image.shape
    for i in range(1, dim1-1):
        for j in range(1, dim2-1):
            image[i,j] = (image_array[(i-1):(i+2),(j-1):(j+2)]*suanzi).sum()
    #归一化
    image = image * (255.0/np.amax(image))
    return image

suanzi = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

image = Image.open("lena_std.tif").convert("L")
image_array = np.array(image)
image2 = imconv(image_array, suanzi)
plt.subplot(2,1,1)
plt.imshow(image_array,cmap="gray")
plt.axis("off")
plt.subplot(2,1,2)
plt.imshow(image2,cmap="gray")
plt.axis("off")
plt.show()


image_blur = signal.convolve2d(image_array, suanzi1, mode="same")
image2 = signal.convolve2d(image_blur, suanzi2, mode="same")
image2 = (image2/float(image2.max()))*255
image2[image2>image2.mean()] = 255
'''


import tensorflow as tf
import glob
from  itertools import groupby
from collections import defaultdict

# 读取所有种类文件夹下所有图片的文件名
image_filenames = glob.glob("./Images/n02*/*.jpg")
temp = image_filenames[0].split("/")
# training_dataset是一个dictionary，key目前缺失，但是value是list，而且初始值是list的初始值[]
training_dataset = defaultdict(list)
testing_dataset = defaultdict(list)

# map函数：对可迭代对象中的每个元素执行func函数，结果作为list返回，如果可迭代对象有多个，则并行处理
# lambda：定义了一个匿名函数，lambda x: x+1 表示参数为x，返回x+1。为了代码更简洁
image_filename_with_breed = map(lambda filename: (filename.split("/")[2], filename),image_filenames)
# itertools.groupby(iterable,[key]) 按照key对iterable进行分组
for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x:x[0]):
    #枚举每个品种的图像，把大约20%的图像分进测试组
    for i, breed_image in enumerate(breed_images):
        if i % 5 == 0:
            testing_dataset[dog_breed].append(breed_image[1])
        else:
            training_dataset[dog_breed].append(breed_image[1])

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)


def write_records_file(dataset, record_location):
    '''
    将dataset中的jpg文件转换为TFRecord文件
    :param dataset（list）: 
    :param record_location: TFRecord文件的存储路机构
    :return: 
    '''
    writer = None
    current_index = 0
    #枚举dataset，每隔100幅图像，训练样本的信息就被写到一个新的TFRecord文件中
    for breed, image_filenames in dataset.items():
        # breed和image_filenames是dataset中的key-value对，image_filenames是一个list，包含一个品种下的所有图片名
        for image_filename in image_filenames:
            # 对image_filenames中的每个文件名进行迭代
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                # 每隔100张图片就写进一个TFRecord文件
                record_filename = "{record_location}-{current_index}.tfrecords".format(record_location = record_location,current_index = current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1
            image_file = tf.read_file(image_filename)

            #在ImageNet的这些狗的图像中，有少量图片是不能被TensorFlow识别成JPEG图像的，利用try/catch将这些图片忽略
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print image_filename
                continue

            # 转换为灰度图可以减少处理的计算量和内存，但这不是必须的
            grayscale_image = tf.image.rgb_to_grayscale(image)
            # 裁剪图像
            resized_image = tf.image.resize_images(grayscale_image,[250,151])

            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()

            #将标签转换为字符串存储较高效，推荐的做法是将其转换为整数索引或读热编码的秩1张量
            image_label = breed.encode("utf-8")
            example = tf.train.Example(features=tf.train.Features(feature={
                                    'label':
                      tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                                    'image':
                      tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
                                }))

            writer.write(example.SerializeToString())
            writer.close()

#write_records_file(testing_dataset, "./output/testing-images/testing-image")
#write_records_file(training_dataset,"./output/training-images/training-image")

# 测试集和训练集已经被转换为TFRecord格式，可以按照TFRecord文件而非JPEG图片进行读取
# 现在的目标是每次加载少量的图片和相应的标签

file_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./output/training-images/*.tfrecords"))
reader = tf.TFRecordReader()
_, serialized = reader.read(file_queue)

features = tf.parse_single_example(
    serialized,
    features={
        'label':tf.FixedLenFeature([], tf.string),
        'image':tf.FixedLenFeature([],tf.string),
    })
record_image = tf.decode_raw(features['image'], tf.uint8)

# 修改图像的形状有助于训练和输出的可视化
image = tf.reshape(record_image, [250, 151, 1])

label = tf.cast(features['label'], tf.string)
min_after_dequeue = 10
batch_size = 3
capacity = min_after_dequeue + 3*batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [image, label],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue
)

#---------开始搭建CNN模型-----------

#将图像转换为灰度值位于【0，1）的浮点类型，以与convolution2d期望的输入匹配
float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)
conv2d_layer_one = tf.contrib.layers.convolution2d(
    float_image_batch,
    num_outputs = 32, #要生成的滤波器数量
    kernel_size = (5,5),
    activation_fn = tf.nn.relu,
 #   weights_initializer = lambda i, dtype, partition_info=None: tf.truncated_normal([38912, 512], stddev=0.1),
    stride = (2, 2),
    trainable = True
)
pool_layer_one = tf.nn.max_pool(
    conv2d_layer_one,
    ksize=[1,2,2,1],
    strides=[1,2,2,1],
    padding='SAME'
)

print conv2d_layer_one.get_shape(), pool_layer_one.get_shape()

conv2d_layer_two = tf.contrib.layers.convolution2d(
    pool_layer_one,
    num_outputs = 64,
    kernel_size = (5,5),
    activation_fn = tf.nn.relu,
    stride = (1,1),
    trainable = True
)

pool_layer_two = tf.nn.max_pool(
    conv2d_layer_two,
    ksize = [1,2,2,1],
    strides = [1,2,2,1],
    padding='SAME'
)

print conv2d_layer_two.get_shape(), pool_layer_two.get_shape()

flattened_layer_two = tf.reshape(
    pool_layer_two,
    [
        batch_size,
        -1
    ]
)

print flattened_layer_two.get_shape()

hidden_layer_three = tf.contrib.layers.fully_connected(
    flattened_layer_two,
    512,
    weights_initializer = lambda i,dtype, partition_info=None: tf.truncated_normal([38912, 512],stddev=0.1),
    activation_fn = tf.nn.relu
)

hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

final_fully_connected = tf.contrib.layers.fully_connected(
    hidden_layer_three,
    120,
    weights_initializer = lambda i, dtype, partition_info=None: tf.truncated_normal([512, 120],stddev=0.1)
)

print final_fully_connected.get_shape()

labels = list(map(lambda c: c.split("/")[-1], glob.glob("./Images/*")))
train_labels = tf.map_fn(lambda l: tf.where(tf.equal(labels, l))[0,0:1][0], label_batch, dtype=tf.int64)

loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(final_fully_connected, train_labels)
)

batch = tf.Variable(0)
learning_rate = tf.train.exponential_decay(
    0.01,
    batch * 3,
    120,
    0.95,
    staircase=True
)

optimizer = tf.train.AdamOptimizer(
    learning_rate,0.9).minimize(
    loss, global_step=batch
)

train_prediction = tf.nn.softmax(final_fully_connected)

sess.run(train_prediction)
pass




