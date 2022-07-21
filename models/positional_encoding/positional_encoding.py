
import tensorflow as tf
import numpy as np

def get_angles(pos, i, model_dim):
    angle_dropout_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(model_dim))
    return pos * angle_dropout_rates

def positional_encoding(position, model_dim):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(model_dim)[np.newaxis, :],
                            model_dim)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

