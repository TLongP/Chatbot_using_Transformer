import tensorflow as tf
from .beamsearch import find_top_k
from models.custom_metrics.metrics import create_prediction_mask
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




class Translator:
    def __init__(self,vectorizer, model, beam_width=3):
        self.vectorizer = vectorizer
        self.model = model
        self.vocabulary = self.vectorizer.get_vocabulary()
        self.beam_width = beam_width
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
        # start array eg. [[2]]
        start_array = tf.TensorArray(
                            dtype=tf.int64, 
                            size=0, 
                            dynamic_size=True) # can change shape
        start_array = start_array.write(0, start_ind)
        
        #  a list of the top k sentence, not sorted
        top_k_sentence = tf.TensorArray(dtype=tf.int64, 
                                        size=0, 
                                        dynamic_size=True,
                                        infer_shape=False,
                                        clear_after_read=False) #set this true next time
        top_k_sentence = top_k_sentence.write(0,start_array.stack())
        
        # a list from predicted top k sentence
        choose_next_top_k = tf.TensorArray(dtype=tf.int64, 
                                        size=0, 
                                        dynamic_size=True,
                                        infer_shape=False,
                                        clear_after_read=False)
        # a list of losses from choose_next_top_k
        # base on the top k from this set we can choose
        # top k from choose_next_top_k
        choose_next_top_loss = tf.TensorArray(dtype=tf.float32, 
                                        size=0, 
                                        dynamic_size=True,
                                        infer_shape=False,
                                        clear_after_read=False)
                                    
        for i in tf.range(1):
            for j in range(top_k_sentence.size()):
                output = tf.transpose(top_k_sentence.read(j))
                predictions, _ = self.model([encoder_input, output], training=False)
                # select the last token from the, eg the next word for the sentence
                predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
                next_values, next_words = find_top_k(predictions) # (values,indexes)
                next_words = tf.cast(next_words,dtype=tf.int64)


                #for each predictions (eg. sentence) we write into choose_next_top_k
                
                for k in range(self.beam_width):
                    # concat the word we which we predicted to the sentence we have
                    new_sentence =tf.concat([output,next_words[...,k]],axis=1)
                    choose_next_top_k.write(j*self.beam_width+k,new_sentence)
                    # next we compute the log loss for this sentence
                    sentence_value,_ =  self.model([encoder_input, new_sentence], training=False)
                    step_value = _compute_sum_log_value(new_sentence,sentence_value)
                    choose_next_top_loss.write(j*self.beam_width+k,step_value)

  
        return top_k_sentence



        
def _compute_sum_log_value(new_sentence,sentence_value):
    mask = create_prediction_mask(new_sentence)
    #sentence_value = tf.math.multiply(sentence_value,mask)
    print(f"sentence {sentence_value.shape}")
    print(f"mask {mask.shape}")
    return 0



