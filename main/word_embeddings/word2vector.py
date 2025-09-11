import re
import tensorflow.keras as k 
import numpy as np 
def word_to_vector():
    with open('text', 'r') as f:
        text = f.read()
        # text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
    
    vocab = set(words)
    
    def genrate_pairs(words, window_size=2):
        pairs = []
        for i, center_word in enumerate(words):
            for j in range(-window_size, window_size):
                context_j = i + j
                if context_j < 0 or context_j == i or context_j >= len(words):
                    continue
                pairs.append((center_word, words[context_j]))
        return pairs
    pairs = genrate_pairs(words)
    word2idx = {word:idx for idx, word in enumerate(vocab)}
    x_train = [word2idx[x] for x, _ in pairs]
    y_train = [word2idx[y] for _, y in pairs]
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    input_data = k.Input(shape=(1, ))
    x = k.layers.Embedding(input_dim=len(vocab), output_dim=10, input_length=1)(input_data)
    x = k.layers.Flatten()(x)
    x  = k.layers.Dense(len(vocab), activation='softmax')(x)

    model = k.models.Model(inputs=input_data, outputs=x)
    model.compile(loss=k.losses.sparse_categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    early_stop = k.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    model.fit(x_train, y_train, epochs=100, callbacks=[early_stop])
    return model, word2idx
