import tensorflow.keras as k 

from inception import bulding_inception
def google_netowrk():

    input_layer = k.Input(shape=(224, 224, 3))
    conv_1 = k.layers.Conv2D(64, kernel_size=(7, 7), strides=(2,2),
                             activation='relu', padding='same')(input_layer)
    
    max_pool = k.layers.MaxPool2D((3,3), strides=(2,2), padding='same')(conv_1)

    x = k.layers.Conv2D(192, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', activation='relu')(max_pool)
    
    x = k.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
    x = bulding_inception(x, [64, 96, 128, 16, 32, 32])
    x = bulding_inception(x, [128, 128, 192, 32, 96, 64])
    x = k.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
    x = bulding_inception(x, [192, 96, 208, 16, 48, 64])
    x = bulding_inception(x, [160, 112, 224, 24, 64, 64])
    x = bulding_inception(x, [128, 128, 256, 24, 64, 64])
    x = bulding_inception(x, [112, 144, 288, 32, 64, 64])
    x = bulding_inception(x, [256, 160, 320, 32, 128, 128])
    x = k.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(x)
    x = bulding_inception(x, [256, 160, 320, 32, 128, 128])
    x = bulding_inception(x, [384, 192, 384, 48, 128, 128])
    x = k.layers.AveragePooling2D((7, 7), strides=(1, 1), padding='same')(x)
    x = k.layers.Dropout(0.4)(x)
    x = k.layers.Dense(1000, activation='softmax')(x)
    model = k.models.Model(inputs=input_layer, outputs=x)
    return model



    