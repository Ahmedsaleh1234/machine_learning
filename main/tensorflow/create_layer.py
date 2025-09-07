import tensorflow as tf

def create_layer(prev, n, activati):
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    y = tf.keras.layers.Dense(n, activati, kernel_initializer=init, name='layer')
    return y(prev)