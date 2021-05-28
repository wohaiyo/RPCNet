import tensorflow as tf
import config as cfg
def cross_entropy_loss(logits, annotation):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                         labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                         name="entropy"))
    return loss

def weighted_cross_entropy_loss_artdeco(decode_logits, binary_label):
    decode_logits_reshape = tf.reshape(
        decode_logits,
        shape=[decode_logits.get_shape().as_list()[0],
               decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
               decode_logits.get_shape().as_list()[3]])

    binary_label_reshape = tf.reshape(
        binary_label,
        shape=[binary_label.get_shape().as_list()[0],
               binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
    binary_label_reshape = tf.one_hot(binary_label_reshape, depth=cfg.DATASET_NUM_CLASSESS)
    class_weights = tf.constant([[0., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])      # exclude outlier
    weights_loss = tf.reduce_sum(tf.multiply(binary_label_reshape, class_weights), 2)
    binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_label_reshape,
                                                               logits=decode_logits_reshape,
                                                               weights=weights_loss)
    binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

    return binary_segmentation_loss

def weighted_cross_entropy_loss_ecp(decode_logits, binary_label):
    decode_logits_reshape = tf.reshape(
        decode_logits,
        shape=[decode_logits.get_shape().as_list()[0],
               decode_logits.get_shape().as_list()[1] * decode_logits.get_shape().as_list()[2],
               decode_logits.get_shape().as_list()[3]])

    binary_label_reshape = tf.reshape(
        binary_label,
        shape=[binary_label.get_shape().as_list()[0],
               binary_label.get_shape().as_list()[1] * binary_label.get_shape().as_list()[2]])
    binary_label_reshape = tf.one_hot(binary_label_reshape, depth=cfg.DATASET_NUM_CLASSESS)
    class_weights = tf.constant([[0., 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])      # exclude outlier
    weights_loss = tf.reduce_sum(tf.multiply(binary_label_reshape, class_weights), 2)
    binary_segmentation_loss = tf.losses.softmax_cross_entropy(onehot_labels=binary_label_reshape,
                                                               logits=decode_logits_reshape,
                                                               weights=weights_loss)
    binary_segmentation_loss = tf.reduce_mean(binary_segmentation_loss)

    return binary_segmentation_loss
