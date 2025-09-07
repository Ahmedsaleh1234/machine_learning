import tensorflow.keras as k

def identity_block(x, filters):
    identity = x
    f_1x1, f_3x3, f2_1x1 = filters
    x = k.layers.Conv2D(f_1x1, kernel_size=(1,1),kernel_initializer='he_normal', 
                        padding='valid')(x)
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)

    x = k.layers.Conv2D(f_3x3, kernel_size=(3,3), kernel_initializer='he_normal',
                        padding='same')(x)
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Conv2D(f2_1x1, kernel_size=(1, 1), strides=(1, 1), kernel_initializer='he_normal',
                        padding='valid')(x)
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Add()([x, identity])
    x = k.layers.Activation('relu')(x)
    return x
    
