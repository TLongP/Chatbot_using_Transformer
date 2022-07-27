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
        next_top_k_sentence = tf.TensorArray(dtype=tf.int64, 
                                        size=0, 
                                        dynamic_size=True,
                                        infer_shape=False,
                                        clear_after_read=False)
        # a list of losses from next_top_k_sentence
        # base on the top k from this set we can choose
        # top k from next_top_k_sentence
        top_k_loss_vector = tf.TensorArray(dtype=tf.float32, 
                                        size=0, 
                                        dynamic_size=True,
                                        infer_shape=False,
                                        clear_after_read=False)


        next_top_loss_vector = tf.TensorArray(dtype=tf.float32, 
                                        size=0, 
                                        dynamic_size=True,
                                        infer_shape=False,
                                        clear_after_read=False)
        next_top_loss = tf.TensorArray(dtype=tf.float32, 
                                        size=0, 
                                        dynamic_size=True,
                                        infer_shape=True,
                                        clear_after_read=False)                     
        for i in tf.range(max_length):



            for j in range(top_k_sentence.size()):

                
                output = top_k_sentence.read(j)
                predictions, _ = self.model([encoder_input, output], training=False)
                # select the last token from the, eg the next word for the sentence
                predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
                next_values, next_words = find_top_k(predictions) # (values,indexes)
                next_words = tf.cast(next_words,dtype=tf.int64)
                output = top_k_sentence.read(j)
                predictions, _ = self.model([encoder_input, output], training=False)
                # select the last token from the, eg the next word for the sentence
                predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
                #  next_values do not lie beetween [0,1] since we do not use soft max at last input
                predictions = tf.nn.softmax(predictions,axis=-1)
                next_values, next_words = find_top_k(predictions,self.beam_width) # (values,indexes)
                next_words = tf.cast(next_words,dtype=tf.int64)
                for k in range(self.beam_width):
                    if i==0:
                        top_k_sentence.write(k,tf.concat([output,next_words[...,k]],axis=1))
                        top_k_loss_vector.write(k,tf.concat([[[0]],next_values[...,k]],axis=1))
                    else:
                        pre_step_loss = top_k_loss_vector.read(j)
                        new_value = tf.concat([pre_step_loss,next_values[...,k]],axis=1)
                        next_top_loss_vector.write(j*self.beam_width+k,new_value)

                        new_sentence = tf.concat([output,next_words[...,k]],axis=1)
                        next_top_k_sentence.write(j*self.beam_width+k,new_sentence)


            if i!=0:
                for l in range(next_top_loss_vector.size()):
                    sentence_value = _compute_sum_log_value(next_top_k_sentence.read(l),
                                                            next_top_loss_vector.read(l))
                    next_top_loss.write(l,sentence_value)

                # now we select the top k value from next_top_loss
                _, top_k_indexes = tf.math.top_k(next_top_loss.stack(),k=self.beam_width)

                for i, top_k_index in enumerate(top_k_indexes):
                    top_k_sentence.write(i,next_top_k_sentence.read(top_k_index))
                    top_k_loss_vector.write(i,next_top_loss_vector.read(top_k_index))
            # if end in all of top_k_sentence we break
            if tf.math.reduce_all(tf.math.reduce_any(tf.where(next_top_k_sentence.stack()==3,True,False),axis=-1)):
                break


  
        return top_k_sentence,next_top_k_sentence



        
def _compute_sum_log_value(new_sentence,sentence_vec):
    mask = create_prediction_mask(new_sentence)
    mask = tf.cast(mask,dtype=tf.float32)
    sentence_value = tf.math.log(tf.math.multiply(sentence_vec,mask))
    return tf.math.reduce_sum(sentence_value)/tf.math.reduce_sum(mask)



