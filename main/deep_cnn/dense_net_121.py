import tensorflow.keras as k

import dense_block
from transalation_layer import transaltion_layer

from dense_block import desnse_block

def dense__net_121(growth_rate=32, comperision=1.0):
    input_layer = k.Input(shape=(224, 224, 3))
    np_filters = 64
    layers = [6, 12, 24, 16]
    x = k.layers.BatchNormalization(axis=3)(input_layer)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Conv2D(np_filters, kernel_size=(7, 7), kernel_initializer='he_normal',
                        strides=(2, 2), padding='same')(x)
    x = k.layers.MaxPool2D((3,3), strides=(2 ,2), padding='same')(x)
    for i in range(len(layers) - 1):
        x, np_filters = desnse_block(x , np_filters, growth_rate, layers[i])
        x, np_filters = transaltion_layer(x, np_filters, comperision)
    x, np_filters = desnse_block(x, np_filters, growth_rate, layers[-1])

    x = k.layers.AveragePooling2D((7, 7))(x)
    x = k.layers.Dense(1000, activation='softmax')(x)

    model = k.models.Model(inputs=input_layer, outputs=x)
    return model

    