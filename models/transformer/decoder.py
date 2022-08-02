import tensorflow as tf
from models.positional_encoding.positional_encoding import  PositionalEncoding
from models.custom_blocks.layers import DecoderLayer


class Decoder(tf.keras.layers.Layer):
    def __init__(self,*, num_layers, model_dim, num_heads, dff, target_vocab_size,
                dropout_rate=0.1, max_tokens=128):
        super().__init__()

        self.model_dim = model_dim
        self.num_layers = num_layers
        self.target_vocab_size = target_vocab_size
        self.max_tokens = max_tokens
        self.dropout_rate = dropout_rate
        self.dff = dff
        self.num_heads = num_heads

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, 
                                                    model_dim)

        self.pos_encoding = PositionalEncoding(max_tokens, model_dim)
        self.dec_layers = [
            DecoderLayer(model_dim=model_dim, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, enc_output, training,
            look_ahead_mask, padding_mask):

        #seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, model_dim)
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))
        
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                    look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2

        return x, attention_weights
    
    def _load_embedding_from_array(self,embedding_array):
        """
        load the embedding array
        """
        self.embedding = tf.keras.layers.Embedding(
            self.target_vocab_size, 
            self.model_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_array),
            trainable=False
                                                    )

    
    def get_config(self):
        config = super().get_config()
        config.update(
            {
            "num_layers" : self.num_layers,
            "model_dim" : self.model_dim,
            "num_heads" : self.num_heads,
            "dff" : self.dff,
            "tartget_vocab_size" : self.target_vocab_size,
            "dropout_rate" : self.dropout_rate,
            "max_tokens" : self.max_tokens
            }
            )
        return config
