from sklearn.feature_extraction.text import CountVectorizer

def bag_words(sentence, vocab=None):
    vector = CountVectorizer(vocabulary=vocab)
    x = vector.fit_transform(sentence)

    return (x, vector.get_feature_names_out())