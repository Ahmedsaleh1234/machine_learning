import create_layer
import placeholder
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
x, y = placeholder.create_placeholders(780, 10)

l = create_layer.create_layer(x, 256, tf.nn.tanh)
print(l)