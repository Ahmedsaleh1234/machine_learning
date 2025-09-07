import tensorflow.compat.v1 as tf


def create_placeholders(nx, classess):
    x = tf.placeholder(tf.float32, shape = (None, nx))
    y = tf.placeholder(tf.float32, shape=(None, classess))

    return x, y