from keras import activations
import tensorflow
import keras as k


def lenet(x):

    conv1 = k.layers.Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', 
                            kernel_initializer='he_normal')(x)
    
    max_pool1 = k.layers.MaxPool2D((2, 2), strides=(2, 2))(conv1)

    conv2 = k.layers.Conv2D(16, kernel_size=(5, 5), padding='valid', activation='relu'
                            , kernel_initializer='he_normal')(max_pool1)
    
    max_pool2 = k.layers.MaxPool2D((2, 2), strides=(2,2))(conv2)
    flatten = k.layers.Flatten()(max_pool2)

    fc1 = k.layers.Dense(120, activation='relu', kernel_initializer='he_normal')(flatten)
    fc2 = k.layers.Dense(84, activation='relu', kernel_initializer='he_normal')(fc1)
    output = k.layers.Dense(10, activation='softmax', kernel_initializer='he_normal')(fc2)

    model = k.Model(inputs=x, outputs=output)
    model.compile(optimizer='adam', loss=k.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model

