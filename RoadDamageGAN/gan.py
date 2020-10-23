#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 13:57:32 2019

@author: yya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:31:42 2019

@author: siat-x
"""

import os
import re
import math
import numpy as np
import tensorflow as tf
from datetime import datetime
import cv2
from glob import glob
from ops import *
from utils import *
from tensorflow.data.experimental import prefetch_to_device, shuffle_and_repeat, map_and_batch
from xml.etree import ElementTree


class GAN:

    def __init__(self, sess, args):
        self.model_name = 'GAN'
        self.sess = sess
        self.gan_type = args.gan_type
        self.epoch = args.epoch
        self.decay_epoch = args.decay_epoch
        self.save_freq = args.save_freq
        self.iteration = args.iteration
        self.init_lr = args.lr
        self.beta1 = 0.5
        
        """ Dir """
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.summaries_dir = args.summaries_dir

        """ Weight """
        self.rec_weight = args.rec_weight
        self.cls_weight = args.cls_weight
        self.ld = args.ld
        
        """ Channels """
        self.batch_size = args.batch_size
        self.genn = args.n_gen
        self.disn = args.n_dis
        self.img_size = args.img_size
        self.img_ch = args.img_ch
        self.img_shape = (args.img_size, args.img_size, args.img_ch)
        self.ch = args.ch
            
        """ Dataset """
        self.label_size = args.label_size
        self.augment_flag = args.augment_flag
        self.train_dataset_type = args.dataset_type
        self.train_dataset_img_type = args.dataset_img_type
        self.train_dataset_path = args.dataset_path
           
        """ Test """
        self.test_path = args.test_path
        
##################################################################################
# Generator
##################################################################################
        
    def generator(self, x, c, reuse=False, train=True):
        ys = []
        channel = 256
        norm='batch'
        with tf.variable_scope('generator', reuse=reuse):

            c = tf.cast(tf.reshape(c, shape=[-1, 1, 1, c.shape[-1]]), tf.float32)
            c = tf.tile(c, [1, x.shape[1], x.shape[2], 1])
            x = tf.concat([x, c], axis=-1)
            x = conv(x, channel, kernel=3, stride=1, pad=1, activation_fn='relu', is_training=train, norm_fn=norm, use_bias=False, scope='conv_b')
            
            for i in range(0, self.genn-2) :
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel, kernel=5, stride=1, pad=2, activation_fn='relu', is_training=train, norm_fn=norm, use_bias=False, scope='deconv_'+str(i))

            for i in range(self.genn-2, self.genn) :
                channel = channel/2
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel, kernel=5, stride=1, pad=2, activation_fn='relu', is_training=train, norm_fn=norm, use_bias=False, scope='deconv_'+str(i))
                
            x = conv(x, self.img_ch, kernel=1, stride=1, pad=0, activation_fn='tanh', is_training=train, norm_fn='None', use_bias=False, scope='conv_x')
            
            return x
        
##################################################################################
# Discriminator
##################################################################################

    def discriminator(self, x, reuse=False, train=True, dropout=False):
        norm='None'#'spectral#instance
        with tf.variable_scope("discriminator", reuse=reuse) :
        
            channel = self.ch
            for i in range(0, self.disn):
                x = conv(x, channel, kernel=4, stride=2, pad=1, activation_fn='leaky', 
                         is_training=train, norm_fn=norm, scope='conv_' + str(i))
                channel = channel * 2 if channel<512 else 512

            logit =  conv(x, 1, 3, stride=1, pad=1, activation_fn='None', 
                          is_training=train, norm_fn='None', use_bias=False, scope='D_logit')

            x = conv(x, self.label_size, kernel=int(x.shape[1]), stride=1, pad=0, activation_fn='None', 
                     is_training=train, norm_fn='None', use_bias=False, scope='D_label')
            
            x = tf.reshape(x, shape=[-1, self.label_size])

            return logit, x
        
##################################################################################
# Model
##################################################################################

    def gradient_panalty(self, real, fake, logit_real, logit_fake, scope="discriminator"):

        if self.gan_type == 'wgan-div':
            grad_real = tf.gradients(logit_real, real)[0]
            grad_fake = tf.gradients(logit_fake, fake)[0]
            grad_real_norm = tf.norm(tf.layers.flatten(grad_real), axis=1) # l2 norm
            grad_fake_norm = tf.norm(tf.layers.flatten(grad_fake), axis=1) # l2 norm
            GP = tf.reduce_mean(tf.pow(grad_real_norm, 3)+tf.pow(grad_fake_norm, 3))
            return GP
            
        if self.gan_type == 'dragan' :
            shape = tf.shape(real)
            eps = tf.random_uniform(shape=shape, minval=0., maxval=1.)
            x_mean, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region
            noise = 0.5 * x_std * eps  # delta in paper
            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=-1., maxval=1.)
            interpolated = tf.clip_by_value(real + alpha * noise, -1., 1.)  # x_hat should be in the space of X
        else :
            shape = tf.shape(real)
            alpha = tf.random_uniform(shape=[shape[0], 1, 1, 1], minval=0., maxval=1.)
            interpolated = alpha*real + (1. - alpha)*fake

        logit, _ = self.discriminator(interpolated, reuse=True)

        GP = 0

        grad = tf.gradients(logit, interpolated)[0] # gradient of D(interpolated)
        grad_norm = tf.norm(tf.layers.flatten(grad), axis=1) # l2 norm

        # WGAN - LP
        if self.gan_type == 'wgan-lp' :
            GP = tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        if self.gan_type == 'wgan-gp' or self.gan_type == 'dragan':
            GP = tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def optimizer_graph_generator(self, gen_loss, dis_loss, learning_rate_g, learning_rate_d, beta1):
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        # optimizer
        gen_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_g,beta1=beta1,beta2=0.999).minimize(gen_loss, var_list=gen_vars)
        dis_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_d,beta1=beta1,beta2=0.999).minimize(dis_loss, var_list=dis_vars)
        return gen_optimizer, dis_optimizer
    
    def build_model(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.input = tf.placeholder(tf.float32, shape=[None, 4, 4, 64], name='input')
        self.lr_g = tf.placeholder(tf.float32, name='lr_g')
        self.lr_d = tf.placeholder(tf.float32, name='lr_d')
        """ Dataset """
        self.Image_Data = ImageData(self.train_dataset_path, img_shape = self.img_shape, augment_flag = self.augment_flag, 
                                    data_type = self.train_dataset_type, img_type = self.train_dataset_img_type, label_size=self.label_size)

        trainA = tf.data.Dataset.from_tensor_slices((self.Image_Data.train_dataset, self.Image_Data.train_label))

        dataset_num = len(self.Image_Data.train_dataset)
        gpu_device = '/gpu:0'
        trainA = trainA.\
            apply(shuffle_and_repeat(dataset_num)).\
            apply(map_and_batch(self.Image_Data.image_processing, self.batch_size, num_parallel_batches=8, drop_remainder=True)).\
            apply(prefetch_to_device(gpu_device, self.batch_size))

        trainA_iterator = trainA.make_one_shot_iterator()

        self.real_imgs, self.label_o = trainA_iterator.get_next()
        
        """ generation """
        self.fake_imgs = self.generator(self.input, self.label_o)
                
        """ Discriminator for real """
        real_logits, real_label = self.discriminator(self.real_imgs)
        
        """ Discriminator for fake """
        fake_logits, fake_label = self.discriminator(self.fake_imgs, reuse=True)
        
        """ Define Loss """
        if self.gan_type.__contains__('wgan') or self.gan_type == 'dragan' :
            grad_pen = self.gradient_panalty(self.real_imgs, self.fake_imgs, real_logits, fake_logits)
        else :
            grad_pen = 0
        
        g_cls_loss = classification_loss2(logit=fake_label, label=self.label_o)
        d_cls_loss = classification_loss2(logit=real_label, label=self.label_o)
        
        dis_loss = discriminator_loss(self.gan_type, real_logits, fake_logits) + self.ld * grad_pen
        gen_loss = generator_loss(self.gan_type, fake_logits)
        
        D_loss = dis_loss + self.cls_weight * d_cls_loss
        G_loss = gen_loss + self.cls_weight * g_cls_loss
        """ Optimizer """
        D_loss += regularization_loss('discriminator')
        G_loss += regularization_loss('generator')
        self.gen_optimizer, self.dis_optimizer = self.optimizer_graph_generator(G_loss, D_loss, self.lr_g, self.lr_d, self.beta1)
        """ Summaries """
        self.g_summary = summary({G_loss:'G_loss',
                                  gen_loss: 'gen_loss',
                                  g_cls_loss:'g_cls_loss'})
        self.d_summary = summary({D_loss:'D_loss',
                                  dis_loss: 'dis_loss', 
                                  d_cls_loss:'d_cls_loss'})
        
    def train(self):
        
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
    
        # summary writer
        self.writer = tf.summary.FileWriter(os.path.join(self.summaries_dir), self.sess.graph)
        
        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.iteration)
            start_batch_id = checkpoint_counter - start_epoch * self.iteration
            step = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            start_epoch = 0
            start_batch_id = 0
            step = 1
            print(" [!] Load failed...")
        self.variables_count()
                
        lr = self.init_lr
        for epoch in range(start_epoch, self.epoch):
            lr = self.init_lr if epoch < self.decay_epoch else self.init_lr * (self.epoch - epoch) / (self.epoch - self.decay_epoch) # linear decay
    
            for idx in range(start_batch_id, self.iteration):
                input_z = np.random.uniform(-1, 1, size=(self.batch_size,4,4,64))
                train_feed_dict = {self.input : input_z, self.lr_g : lr, self.lr_d : lr}
                    
                _,d_summary_opt = self.sess.run([self.dis_optimizer, self.d_summary], feed_dict = train_feed_dict)
                self.writer.add_summary(d_summary_opt, step)
                if (step-1)%5==0:
                    _,g_summary_opt = self.sess.run([self.gen_optimizer, self.g_summary], feed_dict = train_feed_dict)
                    self.writer.add_summary(g_summary_opt, step)
                step += 1
                    
                if np.mod(idx + 1, self.save_freq) == 0:
                    samples_a, samples_t = self.sess.run([self.real_imgs,self.fake_imgs], feed_dict = train_feed_dict)
                    test_shape = (self.batch_size*self.img_size, self.img_size, self.img_ch)
                    samples_a=np.uint8(127.5*(np.reshape(samples_a,test_shape)+1.0))
                    samples_t=np.uint8(127.5*(np.reshape(samples_t,test_shape)+1.0))
                    sample = np.concatenate([samples_a,samples_t],axis=1)#,samples_p
                    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR) 
                    cv2.imwrite(os.path.join(self.sample_dir,str(step)+'.jpg'), sample)
        
                    print(datetime.now().strftime('%c'), ' epoch:', epoch, ' idx:', idx, '/', self.iteration, ' step:', step)
                    self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name + '.model'), global_step=step)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0
    
            # save model for final step
            self.saver.save(self.sess, os.path.join(self.checkpoint_dir, self.model_name + '.model'), global_step=step)

    def test(self):
        
        self.label = tf.placeholder(tf.float32, [1, self.label_size], name='label')
        self.output = self.generator(self.input, self.label, reuse=True, train=False)
        
        # restore check-point if it exits
        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        class_names = ['D20', 'D10', 'D40', 'D00']
        gov_class = ['Czech', 'India', 'Japan']
        for gov in gov_class:
            file_list = os.listdir( self.test_path + gov + '/annotations/xmls')

            for file in file_list:

                im_name = file.split('.')[0] + '.jpg'

                full_impath =  self.test_path + gov + '/images/' + im_name

                save_path =  self.test_path + gov + '/images_x/'
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                infile_xml = open( self.test_path + gov + '/annotations/xmls/' + file, encoding='utf-8')
                tree = ElementTree.parse(infile_xml)
                root = tree.getroot()
                
                cnt = 0
                flag = 0
                for obj in root.iter('object'):
                    cls_name = obj.find('name').text
                    flag = 0
                    for n in class_names:
                        if cls_name == n:
                            flag+=1
                    if flag==0:
                        continue
                    xmlbox = obj.find('bndbox')
                    xmin = int(xmlbox.find('xmin').text)
                    xmax = int(xmlbox.find('xmax').text)
                    ymin = int(xmlbox.find('ymin').text)
                    ymax = int(xmlbox.find('ymax').text)

                    if xmin>xmax:
                        xmin = int(xmlbox.find('xmax').text)
                        xmax = int(xmlbox.find('xmin').text)

                    if ymin>ymax:
                        ymin = int(xmlbox.find('ymax').text)
                        ymax = int(xmlbox.find('ymin').text)

                    # open image
                    img = cv2.imread(full_impath)

                    label = np.reshape(self.Image_Data.get_class_one_hot(cls_name), [1, self.label_size]) 
                    input = np.random.uniform(-1, 1, size=(1,4,4,64))
                    img_o = self.sess.run([self.output], feed_dict={self.input: input, self.label: label})#
                    img_o = np.uint8(127.5*(np.reshape(img_o,(self.img_size, self.img_size, 3))+1.0))
                    img_o = cv2.cvtColor(img_o, cv2.COLOR_RGB2BGR) 
                    img_o = cv2.resize(img_o, (xmax-xmin, ymax-ymin))
                    width, height, channels = img.shape
                    # img_b = cv2.GaussianBlur(img,(5,5),0)
                    # img[ymin:ymax, xmin:xmax] = img_b[ymin:ymax, xmin:xmax]
                    obj = np.zeros_like(img)
                    obj[ymin:ymax, xmin:xmax] = img_o
                    mask = np.zeros_like(img)
                    mask[ymin:ymax, xmin:xmax] = 255
                    center = (xmin+(xmax-xmin)//2, ymin+(ymax-ymin)//2)
                    img = cv2.seamlessClone(obj, img, mask, center, cv2.MIXED_CLONE)
                if flag>0:
                    save_impath = save_path + im_name
                    cv2.imwrite(save_impath, img)
                    # cv2.imshow('img',mask)
                    # cv2.waitKey(0)
        

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
#        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
        
    def set_value(self, matrix, x, y, val):
        w = int(matrix.get_shape()[0])
        h = int(matrix.get_shape()[1])
        mult_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[x, y]], values=[-1.0], dense_shape=[w, h])) + 1.0
        diff_matrix = tf.sparse_tensor_to_dense(tf.SparseTensor(indices=[[x, y]], values=[val], dense_shape=[w, h]))
        matrix = tf.multiply(matrix, mult_matrix) 
        matrix = matrix + diff_matrix
        return matrix
    
    def variables_count(self):
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('generator')]
        dis_vars = [var for var in train_vars if var.name.startswith('discriminator')]
        print("Generator variables:", np.sum([np.prod(v.get_shape().as_list()) for v in gen_vars]))
        print("Discriminator variables:", np.sum([np.prod(v.get_shape().as_list()) for v in dis_vars]))
        print("Total variables:", np.sum([np.prod(v.get_shape().as_list()) for v in train_vars]))