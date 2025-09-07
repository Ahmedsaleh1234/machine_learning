from sklearn.feature_extraction.text import TfidfVectorizer


def TF_IDf(sentences, vocab=None):
    vector = TfidfVectorizer(vocabulary=vocab)

    x = vector.fit_transform(sentences)

    return x, vector.get_feature_names_out() 