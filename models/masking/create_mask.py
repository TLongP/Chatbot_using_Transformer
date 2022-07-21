import tensorflow as tf



def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32) #0 is forpadding in our case ""

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Create a upper right matrix with values 1
    So to predict the first tokens only the first tokens will be used
    for the second first and second
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

