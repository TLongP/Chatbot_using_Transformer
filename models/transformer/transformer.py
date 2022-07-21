import tensorflow as tf
from models.transformer.encoder import Encoder
from models.transformer.decoder import Decoder
from models.masking.create_mask import create_padding_mask, create_look_ahead_mask

class Transformer(tf.keras.Model):
    """
    this model can be found in https://www.tensorflow.org/text/tutorials/transformer
    """
    def __init__(self,*, num_layers, model_dim, num_heads, dff, input_vocab_size,
                target_vocab_size, dropout_rate=0.1,max_tokens=128):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size =target_vocab_size
        self.dropout_rate = dropout_rate
        self.max_tokens = max_tokens



        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim,
                                num_heads=num_heads, dff=dff,
                                input_vocab_size=input_vocab_size, dropout_rate=dropout_rate, 
                                max_tokens=max_tokens)

        self.decoder = Decoder(num_layers=num_layers, model_dim=model_dim,
                                num_heads=num_heads, dff=dff,
                                target_vocab_size=target_vocab_size, dropout_rate=dropout_rate,
                                max_tokens=max_tokens)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training):
        """
        first compute the encoder output then put and the mask
        then put them in the decoder
        """
        inp, tar = inputs

        padding_mask, look_ahead_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, padding_mask)  # (batch_size, inp_seq_len, model_dim)

        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

    def create_masks(self, inp, tar):
        padding_mask = create_padding_mask(inp)
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return padding_mask, look_ahead_mask



    def get_config(self):
        config = super().get_config()
        config.update(
            {
            "num_layers" : self.num_layers,
            "model_dim" : self.model_dim,
            "num_heads" : self.num_heads,
            "dff" : self.dff,
            "input_vocab_size" : self.input_vocab_size,
            "target_vocab_size" : self.target_vocab_size,
            "dropout_rate" : self.dropout_rate,
            "max_tokens" : self.max_tokens
            }
            )
        return config