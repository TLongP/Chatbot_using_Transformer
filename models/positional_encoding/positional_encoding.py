
import tensorflow as tf
import numpy as np

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, model_dim, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        if model_dim % 2 == 1: model_dim += 1 # model_dim must be even
        p, i = np.meshgrid(np.arange(max_seq_len), np.arange(model_dim // 2))
        pos_emb = np.empty((1, max_seq_len, model_dim))
        pos_emb[0, :, ::2] = np.sin(p / 10000**(2 * i / model_dim)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000**(2 * i / model_dim)).T
        self.positional_embedding = tf.constant(pos_emb.astype(self.dtype))
    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_embedding[:, :shape[-2], :shape[-1]]