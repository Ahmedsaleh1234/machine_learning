import tensorflow.keras as k


def projection_block(x, filters, s=2):
    projec = x 
    f_1x1, f_3x3, f2_1x1 =filters

    x = k.layers.Conv2D(f_1x1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                        kernel_initializer='he_normal')(x)
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Conv2D(f_3x3, kernel_size=(3, 3), kernel_initializer='he_normal', 
                        padding='same', strides=(1,1))(x)
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Conv2D(f2_1x1, kernel_size=(1,1), kernel_initializer='he_normal',
                         strides=(1,1), padding='valid')(x)
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)
    #shortcut path
    projec = k.layers.Conv2D(f2_1x1, kernel_size=(1, 1), strides=(s, s), 
                             kernel_initializer='he_normal', padding='valid')(projec)
    
    projec = k.layers.BatchNormalization(axis=3)(projec)
    #adding them
    x = k.layers.Add()([x , projec])
    x = k.layers.Activation('relu')(x)

    return x
    
