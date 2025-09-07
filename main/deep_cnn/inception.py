import tensorflow.keras as k


def bulding_inception(X, filters):
    f1 = filters[0]
    fr3 = filters[1]
    f3 = filters[2]
    f5r = filters[3]
    f5 = filters[4]
    fm_p =filters[5]
    cnn_1x1 = k.layers.Conv2D(f1, kernel_size=(1, 1), activation='relu', 
                              padding='same')(X)
    
    cnn_3x3_reduction = k.layers.Conv2D(fr3, kernel_size=(1, 1), padding='same', 
                                        activation='relu')(X)
    cnn_3x3 = k.layers.Conv2D(f3, kernel_size=(3, 3), activation='relu', 
                              padding='same')(cnn_3x3_reduction)
    cnn_5x5_reduction = k.layers.Conv2D(f5r, kernel_size=(1, 1), activation='relu',
                                        padding='same')(X)
    cnn_5X5 = k.layers.Conv2D(f5, kernel_size=(5, 5), padding='same', activation='relu'
                              )(cnn_5x5_reduction)
    
    max_pool = k.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same'
                                  )(X)
    
    cnn_pool_1x1 = k.layers.Conv2D(fm_p, kernel_size=(1,1), padding='same', 
                                   activation='relu')(max_pool)
    
    output = k.layers.concatenate([cnn_1x1, cnn_5X5, cnn_3x3, cnn_pool_1x1], axis=3)
    return output
