import tensorflow.keras as k

from identity_block import identity_block
from projection_block import projection_block

def resnet50():
    x_input = k.Input(shape=(224, 224, 3))
    #statge_1
    x = k.layers.Conv2D(64, kernel_size=(7, 7),kernel_initializer='he_normal',
                        strides=(2, 2), padding='same')(x_input)
    
    x = k.layers.BatchNormalization(axis=3)(x)
    x = k.layers.Activation('relu')(x)

    x = k.layers.MaxPool2D((3, 3), strides=(2,2), padding='same')(x)

    #stage_2
    x = projection_block(x, [64, 64, 256], s=1)
    x = identity_block(x, [64, 64, 256])
    x = identity_block(x, [64, 64, 256])

    #statge_3

    x = projection_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])
    x = identity_block(x, [128, 128, 512])

    #statge_4
    x = projection_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])
    x = identity_block(x, [256, 256, 1024])

    #statge_5
    x = projection_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])
    x = identity_block(x, [512, 512, 2048])

    x = k.layers.AveragePooling2D((7, 7), padding='same')(x)
    x = k.layers.Dense(1000, activation='softmax')(x)

    model = k.models.Model(inputs=x_input, outputs=x)
    return model



    