import tensorflow as tf
from tensorflow.contrib import slim
from scipy import misc
import os, random
import numpy as np
from glob import glob
from keras.utils import np_utils
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET

class ImageData:

    def __init__(self, data_path, img_shape=(64,64,1), augment_flag=False, data_type='None', img_type='jpg', pad_flag=False, label_size=8):
        self.data_path = data_path
        self.data_type = data_type
        self.img_shape = img_shape
        self.img_h = img_shape[0]
        self.img_w = img_shape[1]
        self.channels = img_shape[2]
        self.augment_flag = augment_flag
        self.img_type = img_type
        self.pad_flag = pad_flag
        self.label_size = label_size
        self.class_names = os.listdir(self.data_path)
        self.train_dataset = []
        self.train_label = []
        images = []
        for cl_name in self.class_names:
            img_names = os.listdir(os.path.join(self.data_path, cl_name))
            for img_name in img_names:
                self.train_dataset.append(os.path.abspath(os.path.join(self.data_path, cl_name, img_name)))
                hot_cl_name = self.get_class_one_hot(cl_name)
                self.train_label.append(hot_cl_name)
        self.train_label = np.reshape(self.train_label, (len(self.train_label), self.label_size))

    def get_class_one_hot(self, class_str):
        label_encoded = self.class_names.index(class_str)
        label_hot = np_utils.to_categorical(label_encoded, len(self.class_names))
        label_hot = label_hot

        return label_hot

    def image_processing(self, filename, label):
        x = tf.read_file(filename)
        if self.img_type == 'jpg':
            x_decode = tf.image.decode_jpeg(x, channels=self.channels)
        if self.img_type == 'png':
            x_decode = tf.image.decode_png(x, channels=self.channels)
        if self.img_type == 'bmp':
            x_decode = tf.image.decode_bmp(x)
            if self.channels == 1 :
                x_decode = tf.image.rgb_to_grayscale(x_decode)
        img = tf.image.resize_images(x_decode, [self.img_h, self.img_w])
        img = tf.reshape(img, [self.img_h, self.img_w, self.channels])
        img = tf.cast(img, tf.float32) / 127.5 - 1

        if self.augment_flag :
            img = tf.cond(pred=tf.greater_equal(tf.random_uniform(shape=[], minval=0.0, maxval=1.0), 0.5),
                          true_fn=lambda: augmentation(img),
                          false_fn=lambda: img)
        
        return img, label
    
def one_hot(batch_size, mask_size, location):
    l = tf.constant([location])
    m = tf.one_hot(l,mask_size,1.,0.)
    m = tf.tile(m,[batch_size,1])
    return m
    
def load_test_data(image_path, size_h=256, size_w=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size_h, size_w])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def augmentation(image):
    seed = random.randint(0, 2 ** 31 - 1)
    image = tf.image.random_flip_left_right(image, seed=seed)
#    image = tf.image.random_brightness(image,max_delta=0.2)
#    image = tf.image.random_contrast(image, 0.5, 1.5)
#    image = tf.clip_by_value(image,-1.,1.)
#    image = tf.image.random_saturation(image, 0, 0.3)
    return image
											         
def GetAnnotBoxLoc(AnotPath):#AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ObjBndBoxSet={} #以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc=[x1,y1,x2,y2]
        if ObjName in ObjBndBoxSet:
            ObjBndBoxSet[ObjName].append(BndBoxLoc)#如果字典结构中含有这个类别了，那么这个目标框要追加到其值的末尾
        else:
            ObjBndBoxSet[ObjName]=[BndBoxLoc]#如果字典结构中没有这个类别，那么这个目标框就直接赋值给其值吧
    return ObjBndBoxSet

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def summary(tensor_collection, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
    """
    usage:

    1. summary(tensor)

    2. summary([tensor_a, tensor_b])

    3. summary({tensor_a: 'a', tensor_b: 'b})
    """

    def _summary(tensor, name, summary_type=['mean', 'stddev', 'max', 'min', 'sparsity', 'histogram']):
        """ Attach a lot of summaries to a Tensor. """

        if name is None:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            name = re.sub('%s_[0-9]*/' % 'tower', '', tensor.name)
            name = re.sub(':', '-', name)

        with tf.name_scope('summary_' + name):
            summaries = []
            if len(tensor.shape) == 0:
                summaries.append(tf.summary.scalar(name, tensor))
            else:
                if 'mean' in summary_type:
                    mean = tf.reduce_mean(tensor)
                    summaries.append(tf.summary.scalar(name + '/mean', mean))
                if 'stddev' in summary_type:
                    mean = tf.reduce_mean(tensor)
                    stddev = tf.sqrt(tf.reduce_mean(tf.square(tensor - mean)))
                    summaries.append(tf.summary.scalar(name + '/stddev', stddev))
                if 'max' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/max', tf.reduce_max(tensor)))
                if 'min' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/min', tf.reduce_min(tensor)))
                if 'sparsity' in summary_type:
                    summaries.append(tf.summary.scalar(name + '/sparsity', tf.nn.zero_fraction(tensor)))
                if 'histogram' in summary_type:
                    summaries.append(tf.summary.histogram(name, tensor))
            return tf.summary.merge(summaries)

    if not isinstance(tensor_collection, (list, tuple, dict)):
        tensor_collection = [tensor_collection]
    with tf.name_scope('summaries'):
        summaries = []
        if isinstance(tensor_collection, (list, tuple)):
            for tensor in tensor_collection:
                summaries.append(_summary(tensor, None, summary_type))
        else:
            for tensor, name in tensor_collection.items():
                summaries.append(_summary(tensor, name, summary_type))
        return tf.summary.merge(summaries)
