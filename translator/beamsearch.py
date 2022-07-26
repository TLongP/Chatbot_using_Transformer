import tensorflow as tf


def find_top_k(pred_value,k=3):
    """
    find top k value in pred_value
    return a tuple of top k values and indexes
    """
    top_k_words = tf.math.top_k(pred_value,k=k)
    return top_k_words


def write_tensor(x:tf.TensorArray ,value:int):
    """
    append value to x
    """
    output_array = tf.TensorArray(
                            dtype=tf.int64,
                            size=0,
                            dynamic_size=True
                            )
    for i,t in enumerate(x.stack()):
        output_array.write(i,t)
    output_array.write(len(output_array.stack()),value)
    return output_array



def log_loss(pred):
    d_loss = tf.math.log(pred)
    return d_loss



def compute_log_loss(loss,mask):
    return tf.reduce_sum(loss)/tf.reduce_sum(mask)




class OneOfK:

    def __init__(self,end_token=3):
        self.sentence_len = 0
        self.end = False
        self.end_token = end_token
        self.output_sentence = tf.TensorArray(
                            dtype=tf.int64,
                            size=0,
                            dynamic_size=True
                            )
        self.output_vector = tf.TensorArray(
                            dtype=tf.float32,
                            size=0,
                            dynamic_size=True
                            )
        self.loss = 0

    def create_array(self,sentence,output_vector):
        for i,t in enumerate(sentence.stack()):
            self.output_sentence.write(i,t)
        for i,t in enumerate(output_vector.stack()):
            self.output_vector.write(i,t)

    def __call__(self,next_word,next_pred):
        if self.end:
            return

        if next_word == self.end_token:
            self.end = True
        self.output_sentence.write(len(self.output_sentence.stack()),next_word)
        self.output_vector.write(len(self.output_vector.stack()),next_pred)
        self.sentence_len += 1
        self._update_loss()
    
    def _update_loss(self):
        d_loss = tf.math.log(self.output_vector.stack())
        self.loss = tf.reduce_sum(d_loss)/self.sentence_len