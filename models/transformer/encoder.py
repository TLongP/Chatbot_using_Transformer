import tensorflow as tf
from models.positional_encoding.positional_encoding import positional_encoding
from models.custom_blocks.layers import EncoderLayer



class Encoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, model_dim, 
                num_heads, dff, input_vocab_size,
                dropout_rate=0.1,max_tokens=128):
        super().__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, model_dim)
        self.pos_encoding = positional_encoding(max_tokens, self.model_dim)

        self.enc_layers = [
            EncoderLayer(
                model_dim=model_dim, 
                num_heads=num_heads,
                dff=dff, 
                dropout_rate=dropout_rate
                )
            for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, model_dim)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, model_dim)
