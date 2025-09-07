import tensorflow.keras as k
def transaltion_layer(x, np_filters, compression):
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Conv2D(int(np_filters * compression), kernel_initializer='he_normal', 
                        kernel_size=(1, 1), padding='same')(x)
    x = k.layers.AveragePooling2D((2, 2), strides=(2,2))(x)
    return x, int(np_filters * compression)