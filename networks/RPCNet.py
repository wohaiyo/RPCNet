from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim

import config as cfg
from tensorflow.python.ops import nn
import math
from tensorflow.contrib.layers.python.layers import initializers
from libs.resnet import resnet_v1

# Learning Shape Prior
def residual_block_channel(x, channel, name=None):
    res = slim.conv2d(x, channel, [3, 3], scope=name + '_1', padding='SAME')
    res = slim.conv2d(res, channel, [3, 3], scope=name + '_2', padding='SAME', activation_fn=None)
    x = slim.conv2d(x, channel, [1, 1], scope=name + '_down', padding='SAME', activation_fn=None)
    return tf.nn.relu(tf.add(x, res))

def large_kernel1(x, c, k, r, name):
    '''
    large kernel for facade
    :param x:  input feature
    :param c: output channel
    :param k: kernel size
    :param r: rate for conv
    :return:
    '''
    # 1xk + kx1
    row = slim.conv2d(x, c, [1, k], scope=name + '/row', rate=r)
    col = slim.conv2d(x, c, [k, 1], scope=name + '/col', rate=r)
    y = row + col
    return y

def large_kernel2(x, c, k, r, name, is_training=True):
    '''
    large kernel for facade
    :param x:  input feature
    :param c: output channel
    :param k: kernel size
    :param r: rate for conv
    :return:
    '''
    # 1xk + kx1
    left_1 = slim.conv2d(x, c, [1, k], scope=name + '/left_1', rate=r, trainable=is_training)
    left = slim.conv2d(left_1, c, [k, 1], scope=name + '/left_2', rate=r, trainable=is_training)

    right_1 = slim.conv2d(x, c, [k, 1], scope=name + '/right_1',rate=r, trainable=is_training)
    right = slim.conv2d(right_1, c, [1, k], scope=name + '/right_2', rate=r, trainable=is_training)

    y = left + right

    return y

def Recurrent_unet_residual(net, is_training):

    # init channel
    channel = 8

    with tf.variable_scope('Unet'):
        conv1 = slim.repeat(net, 2, slim.conv2d, int(channel*1), [3, 3],
                                trainable=is_training, scope='conv1')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME', scope='pool1')
        conv2 = slim.repeat(pool1, 2, slim.conv2d, int(channel*2), [3, 3],
                            trainable=is_training, scope='conv2')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME', scope='pool2')
        conv3 = slim.repeat(pool2, 2, slim.conv2d, int(channel*4), [3, 3],
                            trainable=is_training, scope='conv3')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='pool3')
        conv4 = slim.repeat(pool3, 2, slim.conv2d, int(channel*8), [3, 3],
                            trainable=is_training, scope='conv4')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME', scope='pool4')
        conv5 = slim.repeat(pool4, 2, slim.conv2d, int(channel * 16), [3, 3],
                            trainable=is_training, scope='conv5')
        pool5 = slim.max_pool2d(conv5, [2, 2], padding='SAME', scope='pool5')

        # Global pooling                                                             # enhance global information
        pool5_shape = pool5.get_shape().as_list()
        img_pooling = tf.reduce_mean(pool5, [1, 2], name='image_level_global_pooling', keep_dims=True)
        img_pooling = slim.conv2d(img_pooling, int(channel * 16), [1, 1], scope='image_level_conv_1x1',
                                  activation_fn=None, trainable=is_training)
        img_pooling = tf.image.resize_bilinear(img_pooling, (pool5_shape[1], pool5_shape[2]))

        lk_pool5 = large_kernel2(pool5, int(channel * 16), 15, 1, 'lk_pool5', is_training=is_training)
        pool5_fuse = img_pooling + lk_pool5
        pool4_shape = pool4.get_shape().as_list()
        pool5_up = tf.image.resize_images(pool5_fuse, [pool4_shape[1], pool4_shape[2]])
        pool5_channel = slim.conv2d(pool5_up, int(channel * 8), [1, 1], scope='pool5_channel', trainable=is_training)

        lk_pool4 = large_kernel2(pool4, int(channel * 8), 15, 1, 'lk_pool4', is_training=is_training)
        pool4_fuse = pool5_channel + lk_pool4
        pool3_shape = pool3.get_shape().as_list()
        pool4_up = tf.image.resize_images(pool4_fuse, [pool3_shape[1], pool3_shape[2]])
        pool4_channel = slim.conv2d(pool4_up, int(channel * 4), [1, 1], scope='pool4_channel', trainable=is_training)

        lk_pool3 = large_kernel2(pool3, int(channel * 4), 15, 1, 'lk_pool3', is_training=is_training)
        pool3_fuse = pool4_channel + lk_pool3
        pool2_shape = pool2.get_shape().as_list()
        pool3_up = tf.image.resize_images(pool3_fuse, [pool2_shape[1], pool2_shape[2]])
        pool3_channel = slim.conv2d(pool3_up, int(channel * 2), [1, 1], scope='pool3_channel', trainable=is_training)

        lk_pool2 = large_kernel2(pool2, int(channel * 2), 15, 1, 'lk_pool2', is_training=is_training)
        pool2_fuse = pool3_channel + lk_pool2
        pool1_shape = pool1.get_shape().as_list()
        pool2_up = tf.image.resize_images(pool2_fuse, [pool1_shape[1], pool1_shape[2]])
        pool2_channel = slim.conv2d(pool2_up, int(channel * 1), [1, 1], scope='pool2_channel', trainable=is_training)

        lk_pool1 = large_kernel2(pool1, int(channel * 1), 15, 1, 'lk_pool1', is_training=is_training)        # enhance details
        pool1_fuse = pool2_channel + lk_pool1
        net_shape = net.get_shape().as_list()
        pool1_up = tf.image.resize_images(pool1_fuse, [net_shape[1], net_shape[2]])

        out = pool1_up

    outputs = slim.conv2d(out, 3, [1, 1],
                             scope='logits', trainable=is_training, activation_fn=None, normalizer_fn=None)
    return outputs

def inference_rpcnet(image, occ, is_training):

    os = 1
    img_shape = image.get_shape().as_list()
    image = tf.image.resize_images(image, [int(img_shape[1] / os), int(img_shape[2] / os)])
    # image = tf.expand_dims(image[:, :, :, 0], 3)    # Range: [0, 1]

    occ = tf.image.resize_images(occ, [int(img_shape[1] / os), int(img_shape[2] / os)])
    occ = tf.expand_dims(occ[:, :, :, 0], 3)

    name_scope = 'rpcnet'
    # Recurrent Block
    input1 = tf.concat([image, occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out1 = Recurrent_unet_residual(input1, is_training)

    input2 = tf.concat([tf.nn.softmax(out1), occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out2 = Recurrent_unet_residual(input2, is_training)

    input3 = tf.concat([tf.nn.softmax(out2), occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out3 = Recurrent_unet_residual(input3, is_training)

    input4 = tf.concat([tf.nn.softmax(out3), occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out4 = Recurrent_unet_residual(input4, is_training)

    input5 = tf.concat([tf.nn.softmax(out4), occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out5 = Recurrent_unet_residual(input5, is_training)

    output1 = tf.image.resize_images(out1, [img_shape[1], img_shape[2]])
    output2 = tf.image.resize_images(out2, [img_shape[1], img_shape[2]])
    output3 = tf.image.resize_images(out3, [img_shape[1], img_shape[2]])
    output4 = tf.image.resize_images(out4, [img_shape[1], img_shape[2]])
    output5 = tf.image.resize_images(out5, [img_shape[1], img_shape[2]])
    label_pred = tf.expand_dims(tf.argmax(tf.nn.softmax(output5), axis=3, name="prediction"), dim=3)

    loss_item = [output1, output2, output3, output4, output5]

    return label_pred, loss_item

def refine_block_window_balcony_residual(x, occ, is_training):

    name_scope = 'rpcnet'
    # Recurrent Block
    input1 = tf.concat([x, occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out1 = Recurrent_unet_residual(input1, is_training)

    input2 = tf.concat([tf.nn.softmax(out1), occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out2 = Recurrent_unet_residual(input2, is_training)

    input3 = tf.concat([tf.nn.softmax(out2), occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out3 = Recurrent_unet_residual(input3, is_training)

    input4 = tf.concat([tf.nn.softmax(out3), occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out4 = Recurrent_unet_residual(input4, is_training)

    input5 = tf.concat([tf.nn.softmax(out4), occ], axis=3)
    with tf.variable_scope(name_scope, reuse=tf.AUTO_REUSE):
        out5 = Recurrent_unet_residual(input5, is_training)


    return out1, out2, out3, out4, out5

def CAM(input_feature, depth=256):
    with tf.variable_scope('cam', reuse=tf.AUTO_REUSE):
        # 1x1 conv
        at_pooling1x1 = slim.conv2d(input_feature, depth, [1, 1], scope='conv1x1', activation_fn=None)

        # rate = 6
        at_pooling3x3_1 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_1', rate=12, activation_fn=None)

        # rate = 12
        at_pooling3x3_2 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_2', rate=24, activation_fn=None)

        # rate = 18
        at_pooling3x3_3 = slim.conv2d(input_feature, depth, [3, 3], scope='conv_3x3_3', rate=36, activation_fn=None)

        # image pooling
        img_pooling = tf.reduce_mean(input_feature, [1, 2], name='image_level_global_pooling', keep_dims=True)
        img_pooling = slim.conv2d(img_pooling, depth, [1, 1], scope='image_level_conv_1x1', activation_fn=None)
        img_pooling = tf.image.resize_bilinear(img_pooling, (input_feature.get_shape().as_list()[1],
                                                             input_feature.get_shape().as_list()[2]))

        net = tf.concat([img_pooling, at_pooling1x1, at_pooling3x3_1, at_pooling3x3_2, at_pooling3x3_3],
        axis = 3, name = 'atrous_concat')
        net = slim.conv2d(net, depth, [1, 1], scope='conv_1x1_output', activation_fn=None)

        net0 = net

        net = large_kernel1(net, 256, 15, 1, 'gcn_a')
        net = net + net0

    return net

def inference_prior_occ(image, is_training):
    img_shape = image.get_shape().as_list()

    # Feature extractor: ResNet50
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, end_points = resnet_v1.resnet_v1_50(image, num_classes=None, is_training=None,
                                                 global_pool=False, output_stride=8,
                                                 spatial_squeeze=False)

    # coarse seg
    net_coarse = net # resudual_block_channel(net, 256, 'net_coarse')


    # context aggregation module
    net_coarse_cam = CAM(net_coarse, depth=256)

    with tf.variable_scope('decoder'):
        # Low level
        low_level_features = end_points['resnet_v1_50/block1/unit_2/bottleneck_v1']
        low_level_features = slim.conv2d(low_level_features, 48, [1, 1], scope='low_level_feature_conv_1x1')
        low_level_features_shape = low_level_features.get_shape().as_list()[1:3]

        # Upsample
        net_coarse = tf.image.resize_images(net_coarse_cam, low_level_features_shape)
        net_coarse = tf.concat([net_coarse, low_level_features], axis=3)
        net_coarse = slim.conv2d(net_coarse, 256, [3, 3], scope='conv_3x3_1')
        net_coarse = slim.conv2d(net_coarse, 256, [3, 3], scope='conv_3x3_2')

    # Classifier
    net_coarse_4s = slim.conv2d(net_coarse, cfg.DATASET_NUM_CLASSESS, [1, 1], scope='logits_coarse',
                              trainable=is_training, activation_fn=None)
    output_net_coarse = tf.image.resize_images(net_coarse_4s, [img_shape[1], img_shape[2]])
    label_pred1 = tf.expand_dims(tf.argmax(output_net_coarse, axis=3, name="prediction"), dim=3)


    # net occlusion
    net_occ = residual_block_channel(net, 256, 'res_net_occ')
    net_occ_8s = slim.conv2d(net_occ, 2, [1, 1], scope='logits_occ', trainable=is_training,
                             activation_fn=None)
    output_net_occ = tf.image.resize_images(net_occ_8s, [img_shape[1], img_shape[2]])
    # occ map
    occ_map = tf.expand_dims(tf.nn.softmax(output_net_occ)[:, :, :, 1], 3)

    # Refine
    net_coarse_8s_softmax = tf.nn.softmax(output_net_coarse)
    win_channel = tf.expand_dims(net_coarse_8s_softmax[:, :, :, cfg.WINDOW_CHANNEL], 3)
    bal_channel = tf.expand_dims(net_coarse_8s_softmax[:, :, :, cfg.BALCONY_CHANNEL], 3)
    bg_channel = 1 - (win_channel + bal_channel)
    # recurrent priornet
    bg_att_window_balcony = tf.concat([bg_channel, win_channel, bal_channel], 3)

    # refine block
    win_bal1, win_bal2, win_bal3, win_bal4, win_bal5  = refine_block_window_balcony_residual(
        bg_att_window_balcony, occ_map, is_training=False)
    bg_att_window_balcony_refine = win_bal2

    return label_pred1, output_net_coarse, output_net_occ, \
           tf.image.resize_images(bg_att_window_balcony, [img_shape[1], img_shape[2]]), \
           tf.image.resize_images(bg_att_window_balcony_refine, [img_shape[1], img_shape[2]])

