import numpy as np
import tensorflow as tf


def load_embedding_array(path,vocab,dim,encoding="utf-8"):
    """
    most of the code can be found here
    https://keras.io/examples/nlp/pretrained_word_embeddings/
    """
    num_tokens = len(vocab)
    word_index = dict(zip(vocab, range(len(vocab))))

    embeddings_index = {}
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs


    for d in embeddings_index.values():
        embedding_dim = len(d)
        break
    assert embedding_dim == dim

    embedding_matrix = np.zeros((num_tokens, dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

