import tensorflow.keras as k
def block_layer(x, np_filters):

    channels = np_filters * 4
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Conv2D(channels, kernel_size=(1, 1), kernel_initializer='he_normal',
                        padding='same')(x)
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)
    x = k.layers.Conv2D(np_filters, kernel_size=(3, 3), kernel_initializer='he_normal',
                        padding='same')(x)
    
    return x

def desnse_block(x, np_filters, growth_rate, block_layers):
    conc_feature = x
    for i in range(block_layers):
        x = block_layer(conc_feature, growth_rate)
        conc_feature = k.layers.concatenate([x, conc_feature], axis=3)
        np_filters += growth_rate
    return conc_feature , np_filters