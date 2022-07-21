import tensorflow as tf
from models.positional_encoding.positional_encoding import positional_encoding, PositionalEncoding
from models.custom_blocks.layers import DecoderLayer



class Decoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, model_dim, num_heads, dff, target_vocab_size,
                dropout_rate=0.1, max_tokens=128):
        super().__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, model_dim)
        #self.pos_encoding = positional_encoding(max_tokens, model_dim)
        self.pos_encoding = PositionalEncoding(max_tokens, model_dim)
        self.dec_layers = [
            DecoderLayer(model_dim=model_dim, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
            look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, model_dim)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        #x += self.pos_encoding[:, :seq_len, :]
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                    look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, model_dim)
        return x, attention_weights
