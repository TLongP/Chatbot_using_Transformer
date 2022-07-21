import tensorflow as tf

class TranslatorWithVectorizer(tf.Module):
    def __init__(self,vectorizer, model):
        self.vectorizer = vectorizer
        self.model = model
        self.vocabulary = self.vectorizer.get_vocabulary()
    def __call__(self, sentence, max_length):
        """
        input will be only 1 sentence
        get a question and return the answer
        to compute the encoder 
        first predict start_ind then use (start_ind,predicted_value) to predict the next word
        note that we use the encoder output to put in the decoder
        """
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        sentence = self.vectorizer(sentence)
        encoder_input = sentence
        start_end = self.vectorizer ([''])[0]  # return start and end index
        start_ind = start_end[0][tf.newaxis]   
        end_ind = start_end[1][tf.newaxis]   
        output_array = tf.TensorArray(
                            dtype=tf.int64, 
                            size=0, dynamic_size=True) # can change shape
        output_array = output_array.write(0, start_ind) # write start ind 
        for i in tf.range(max_length):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.model([encoder_input, output], training=False)
            # select the last token from the seq_len dimension
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)
            # concatentate the predicted_id to the output which is given to the decoder
            # as its input.
            output_array = output_array.write(i+1, predicted_id[0])
            if predicted_id == end_ind: # end if the prediction is end_ind
                break
        output = tf.transpose(output_array.stack())
        _, attention_weights = self.model([encoder_input, output[:,:-1]],
                                                 training=False)
        answer = ""
        for word in tf.squeeze(output):
            answer += self.vocabulary[word] + " "
        return answer, output, attention_weights