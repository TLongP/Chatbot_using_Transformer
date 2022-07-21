import tensorflow as tf
from models.attention.attention import CustomMultiHeadAttention


def point_wise_feed_forward_network(model_dim, dff):

    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(model_dim)  # (batch_size, seq_len, model_dim)
    ])


class FeedForward(tf.keras.layers.Layer):
    """
    create densly connected layer
    this create the feed forward part
    """
    def __init__(self,model_dim,dff,dropout_rate):
        super().__init__()
        self.linear1 = tf.keras.layers.Dense(dff, activation='relu')
        self.linear2 = tf.keras.layers.Dense(model_dim)
        self.dropout_layer1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout_layer2 = tf.keras.layers.Dropout(dropout_rate)
    def call(self,x,training=False):
        x = self.linear1(x)
        x = self.dropout_layer1(x,training=training)
        x = self.linear2(x)
        x = self.dropout_layer2(x,training=training)
        return x



class EncoderLayer(tf.keras.layers.Layer):
    """
    create the 'Nx' layer in the encoder
    """
    def __init__(self,*, model_dim, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.mha = CustomMultiHeadAttention(model_dim=model_dim, num_heads=num_heads)
        self.ffn = FeedForward(model_dim, dff, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    def call(self, x, training, mask):

        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, model_dim)
        attn_output = self.dropout(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, model_dim)

        ffn_output = self.ffn(out1,training=training)  # (batch_size, input_seq_len, model_dim)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, model_dim)

        return out2



class DecoderLayer(tf.keras.layers.Layer):
    """
    create 'Nx' layer in the decoder
    """
    def __init__(self,*, model_dim, num_heads, dff, dropout_rate=0.1):
        super().__init__()

        self.mha1 = CustomMultiHeadAttention(model_dim=model_dim, num_heads=num_heads)
        self.mha2 = CustomMultiHeadAttention(model_dim=model_dim, num_heads=num_heads)

        self.ffn = FeedForward(model_dim, dff, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
            look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, model_dim)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, model_dim)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, model_dim)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, model_dim)

        ffn_output = self.ffn(out2,training=training)  # (batch_size, target_seq_len, model_dim)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, model_dim)

        return out3, attn_weights_block1, attn_weights_block2

