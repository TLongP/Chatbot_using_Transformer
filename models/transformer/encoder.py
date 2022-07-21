import tensorflow as tf
from models.positional_encoding.positional_encoding import PositionalEncoding
from models.custom_blocks.layers import EncoderLayer



class Encoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, model_dim, 
                num_heads, dff, input_vocab_size,
                dropout_rate=0.1,max_tokens=128):
        super().__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.dropout_rate = dropout_rate
        self.max_tokens = max_tokens

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, model_dim)
        self.pos_encoding = PositionalEncoding(max_tokens, model_dim)
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
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, model_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
            "num_layers" : self.num_layers,
            "model_dim" : self.model_dim,
            "num_heads" : self.num_heads,
            "dff" : self.dff,
            "input_vocab_size" : self.input_vocab_size,
            "dropout_rate" : self.dropout_rate,
            "max_tokens" : self.max_tokens
            }
            )
        return config