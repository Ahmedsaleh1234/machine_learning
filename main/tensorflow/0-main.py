#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import placeholder

create_placeholders = placeholder.create_placeholders 
x, y = create_placeholders(784, 10)
print(x)
print(y)