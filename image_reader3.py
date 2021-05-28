import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import tensorflow as tf
import config as cfg
import cv2
from scipy import misc

def image_scaling(img, label, occ):
    """
        Randomly scales the images and labels between minScale and maxScale.
        """

    scale = tf.random_uniform([1], minval=cfg.MIN_SCALE, maxval=cfg.MAX_SCALE, dtype=tf.float32, seed=None)

    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_nearest_neighbor(tf.expand_dims(img, 0), new_shape)
    img = tf.squeeze(img, squeeze_dims=[0])

    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    occ = tf.image.resize_nearest_neighbor(tf.expand_dims(occ, 0), new_shape)
    occ = tf.squeeze(occ, squeeze_dims=[0])

    return img, label, occ


def image_mirroring(img, label, occ):
    """
    Randomly mirrors the images and labels.
    """

    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)

    label = tf.reverse(label, mirror)
    occ = tf.reverse(occ, mirror)

    return img, label, occ


def random_crop_and_pad_image_and_labels(image, label, occ, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images and labels.
    """
    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label  # Needs to be subtracted and later added due to 0 padding.

    occ = tf.cast(occ, dtype=tf.float32)

    combined = tf.concat(axis=2, values=[image, label, occ])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]),
                                                tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]

    combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, 3])
    # split img
    img_crop = combined_crop[:, :, :last_image_dim]
    img_crop = tf.cast(img_crop, dtype=tf.uint8)
    # split label
    label_crop = combined_crop[:, :, last_image_dim: last_image_dim + 1]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    # split occ
    occ_crop = combined_crop[:, :, last_image_dim + 1: last_image_dim + 2]
    occ_crop = tf.cast(occ_crop, dtype=tf.uint8)



    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 1))
    label_crop.set_shape((crop_h, crop_w, 1))
    occ_crop.set_shape((crop_h, crop_w, 1))


    return img_crop, label_crop, occ_crop

def get_image_and_labels(image, label, occ, crop_h, crop_w):
    # Set static shape so that tensorflow knows shape at compile time.

    new_shape = tf.squeeze(tf.stack([crop_h, crop_w]))
    image = tf.image.resize_nearest_neighbor(tf.expand_dims(image, 0), new_shape)
    image = tf.squeeze(image, squeeze_dims=[0])

    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    occ = tf.image.resize_nearest_neighbor(tf.expand_dims(occ, 0), new_shape)
    occ = tf.squeeze(occ, squeeze_dims=[0])

    image.set_shape((crop_h, crop_w, 1))
    occ.set_shape((crop_h, crop_w, 1))
    label.set_shape((crop_h, crop_w, 1))

    return image, label, occ

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
      data_dir: path to the directory with images and masks.
      data_list: path to the file with lines of the form '/path/to/image /path/to/label
                                                        /path/to/mask  /path/to/boundary '.
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    labels = []
    occs = []

    for line in f:
        try:
            image, label, occ  = line.strip("\n").split(' ')
        except ValueError:  # Adhoc for test.
            image = label = occ = line.strip("\n")
        images.append(data_dir + image)
        labels.append(data_dir + label)
        occs.append(data_dir + occ)


    return images, labels, occs


def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, random_crop_pad,
                          ignore_label, img_mean):  # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      random_color: random brightness, contrast, hue and saturation.
      random_crop_pad: random crop and padding for h and w of image
      ignore_label: index of label to ignore during the training.
      img_mean: vector of mean colour values.

    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    occ_contents = tf.read_file(input_queue[2])

    img = tf.image.decode_png(img_contents, channels=1)

    label = tf.image.decode_png(label_contents, channels=1)

    occ = tf.image.decode_png(occ_contents, channels=1)


    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label, occ = image_scaling(img, label, occ)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label, occ = image_mirroring(img, label, occ)

        # Randomly crops the images and labels.
        if random_crop_pad:
            img, label, occ  = random_crop_and_pad_image_and_labels(img, label, occ, h, w, ignore_label)
        else:
            img, label, occ = get_image_and_labels(img, label, occ, h, w)

    # No need to extract mean.
    # img -= img_mean

    return img, label, occ


class ImageReader3(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, input_size, random_scale, random_mirror, random_crop_pad, ignore_label, img_mean, coord):
        '''Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          random_crop_pad: whether to randomly corp and pading images.
          ignore_label: index of label to ignore during the training.
          img_mean: vector of mean colour values.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list, self.occ_list = read_labeled_image_list(self.data_dir, self.data_list)

        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.occs = tf.convert_to_tensor(self.occ_list, dtype=tf.string)


        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.occs],
                                                   shuffle=True)  # not shuffling if it is val
        self.image, self.label, self.occ = read_images_from_disk(self.queue, self.input_size,
                                                                                 random_scale, random_mirror,
                                                                                 random_crop_pad,
                                                                                 ignore_label, img_mean)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.

        Args:
          num_elements: the batch size.

        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        image_batch, label_batch, occ_batch = tf.train.batch(
            [self.image, self.label, self.occ],
            num_elements)
        return tf.cast(image_batch, dtype=tf.int32), tf.cast(label_batch, dtype=tf.int32), tf.cast(occ_batch, dtype=tf.int32)

    # def getqueue(self, num_elements):
    #     '''Pack images and labels into a batch.
    #
    #     Args:
    #       num_elements: the batch size.
    #
    #     Returns:
    #       Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
    #     image_queue = tf.train.batch(
    #         [self.queue],
    #         num_elements)
    #     return image_queue


if __name__ == '__main__':

    input_size = (cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader3(
            cfg.TRAIN_DATA_DIR,
            cfg.TRAIN_DATA_LIST,
            input_size,
            cfg.RANDOM_SCALE,
            cfg.RANDOM_MIRROR,
            cfg.RANDOM_CROP_PAD,
            cfg.IGNORE_LABEL,
            cfg.IMG_MEAN,
            coord)
        image_batch, label_batch, occ_batch = reader.dequeue(cfg.BATCH_SIZE)


    with tf.Session() as se:
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=se)

        imgs, labels, occs = se.run([image_batch, label_batch, occ_batch])

        img = np.array(imgs[0]) * 100
        label = np.squeeze(labels[0], axis=2) * 100
        occ = np.squeeze(occs[0], axis=2)

        cv2.imwrite('test_img5.png', img)
        cv2.imwrite('test_label.png', label)
        cv2.imwrite('test_occ.png', occ)
        coord.request_stop()
        coord.join(threads)
print('Done image reader3')