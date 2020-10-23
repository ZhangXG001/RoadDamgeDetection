#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 20:41:18 2018

@author: yya
"""
import tensorflow as tf
import argparse
from utils import *
from gan import GAN


"""parsing and configuration"""
def parse_args():
    desc = "Tensorflow implementation of SMSD"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    
    parser.add_argument('--dataset_type', type=str, default='CelebA', help='datasetA_type')
    parser.add_argument('--dataset_img_type', type=str, default='png', help='datasetA_image_type')
    parser.add_argument('--dataset_path', type=str, default='./IEEEbigdata2020/data/', help='dataset_path')
    parser.add_argument('--label_size', type=int, default=4, help='label_size')

    parser.add_argument('--epoch', type=int, default=20, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size of each dataset')
    parser.add_argument('--save_freq', type=int, default=200, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')
    parser.add_argument('--rec_weight', type=float, default=5, help='Weight about Reconstruction')
    parser.add_argument('--cls_weight', type=float, default=10, help='Weight about Classification')

    parser.add_argument('--gan_type', type=str, default='wgan-gp', help='gan / lsgan / wgan-gp / wgan-lp / dragan / hinge')
    
    parser.add_argument('--test_path', type=str, default='./IEEEbigdata2020/train/', help='test images path')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_gen', type=int, default=5, help='The number of generator blocks')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layers')

    parser.add_argument('--img_size', type=int, default=128, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='model',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--summaries_dir', type=str, default='summaries',
                        help='Directory name to save training summaries')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.summaries_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = GAN(sess, args)

        # build graph
        gan.build_model()

        if args.phase == 'train' :
            gan.train()
            print(" [*] Training finished!")

        if args.phase == 'test' :
            gan.test()
            print(" [*] Test finished!")

if __name__ == '__main__':
    main()