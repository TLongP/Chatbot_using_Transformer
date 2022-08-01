import tensorflow as tf

class TranslatorWithBeamSearch(tf.Module):
    def __init__(self,encoder,decoder, model, beam_width=10, alpha=0.7):

        self.encoder = encoder
        self.decoder = decoder
        self.model = model
        self.vocabulary = self.decoder.get_vocabulary()
        self.beam_width = beam_width
        self.alpha = alpha
    def __call__(self, sentence, max_length):
        """
        input will be only 1 sentence
        get a question and return the answer
        to compute the encoder 
        first predict start_token then use (start_token,predicted_value)
        to predict the next word
        note that we use the encoder output to put in the decoder
        """

        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        
        sentence = self.encoder(sentence)
        encoder_input = sentence
        start_end = self.decoder([''])[0]  # return start and end index
        start_token = start_end[0][tf.newaxis][tf.newaxis]   # of shape (1,1) 
        end_token = start_end[1] # a number




        # we first put the start token in our top k
        top_k_sentence = start_token 

                 
        for i in tf.range(max_length):
            if i==1:
                # repeat encoder_input beam_width times 
                # so we get a batch size of beam_width 
                # for the encoder
                # only after the first run
                encoder_input = tf.repeat(
                                        encoder_input,
                                        (self.beam_width,),axis=0)
        
            # now take the top k to use for prediction
            output = top_k_sentence
            predictions, _ = self.model([encoder_input, output], 
                                        training=False)

            # take the last word
            predictions = predictions[:, -1:, :]
            predictions = tf.nn.softmax(predictions,axis=-1)

            # choose the top k for each 
            # this will returns for the first vector( [0,...]) 
            # a tuple of k values and k words
            next_k_values, next_k_indexes = tf.math.top_k(
                                                predictions,
                                                k=self.beam_width)

            # next we repeat the output depends on the beam_width
            # so that output is of shape (beam_width,sentence_len) for i=0
            # and for i>0 (beam_width**2,sentence_len)
            # so that we can concatenate the top words we predicted with output
            repeat_tuple = output.shape[0]*(self.beam_width,) 
            output = tf.repeat(output,repeats=repeat_tuple,axis=0)


            next_k_indexes = tf.cast(tf.reshape(next_k_indexes,(-1,1)),
                                        dtype=tf.int64)
            next_k_values = tf.nn.softmax(next_k_values,axis=-1)
            next_k_values = tf.reshape(next_k_values,(-1,1))
            # we can choose top k sentence from next_top_k_sentence
            next_top_k_sentence = tf.concat([output,next_k_indexes],axis=-1)


            if i==0:
                # add 1 to the beggining
                # this will help us to compute the sum log
                ones = tf.ones_like(next_k_values)
                current_k_loss_vectors = tf.concat([ones,next_k_values],axis=-1)
                next_top_loss_vectors = current_k_loss_vectors

            else:
                # next_top_loss_vectors[:,1:] is the associate values which we get if 
                # we use the model to predict next_top_loss_vectors
                # note that the beggining from next_top_loss_vectors is 1

                next_top_loss_vectors = tf.concat([tf.repeat(
                                                    current_k_loss_vectors,
                                                    repeat_tuple,axis=0),
                                                    next_k_values],axis=-1)



            # now select the best top k
            _values = _compute_sum_log_value(next_top_k_sentence,
                                            next_top_loss_vectors,
                                            end_token,
                                            self.alpha)   

            _, new_top_k_words = tf.math.top_k(_values,k=self.beam_width)
            # chossing the top k 
            best_next_sentences = tf.gather(
                        next_top_k_sentence,
                        indices=new_top_k_words)
            best_next_values = tf.gather(
                        next_top_loss_vectors,
                        indices=new_top_k_words)

            top_k_sentence = best_next_sentences
            current_k_loss_vectors = best_next_values
            # if the op k end in every sentence
            # we end since it wont get any better
            if tf.math.reduce_all(tf.math.reduce_any(tf.where(top_k_sentence==end_token,
                                                                True,False),axis=-1)):
                break


        #return the attention_weight for the best sentence
        #_, attention_weight = self.model([encoder_input[0,tf.newaxis], top_k_sentence[0][tf.newaxis]], training=False)
        
        # if you want to use only attention_weights 
        # until the true end of the predicted sentence
        #mask = _create_prediction_mask(top_k_sentence[0],end_token)
        #attention_weights= attention_weights*mask
        return top_k_sentence[0],self._translate(top_k_sentence[0],start_token[0,0],end_token)

    def _translate(self,text,start_token,end_token):
        sentence = [self.vocabulary[word] for word in text if word!=start_token and word!=end_token]
        return " ".join(sentence)


        
def _compute_sum_log_value(new_sentence,sentence_vec,end_token,alpha=0.7):
    """
    compute 1/T^\alpha sum log(P(y_i|x,y_1,...,y_i-1))
    mask: is the true long of the sentence
    so that the sentence ends with the first end_token
    """
    mask = _create_prediction_mask(new_sentence,end_token)
    mask = tf.cast(mask,dtype=tf.float32)
    sentence_value = tf.math.log(tf.math.multiply(sentence_vec,mask))
    return tf.math.reduce_sum(sentence_value,axis=-1)/tf.math.pow(
                            tf.math.reduce_sum(mask,axis=-1),alpha)



def _create_prediction_mask(pred,end_token):
    """
    can be found in models.custom_metrics.metrics
    this will returns a mask with value 1 until the first 
    end token reach 
    all that come after the first end token will be 
    mask with 0
    """
    mask_end_token = tf.cast(tf.math.equal(pred,end_token),dtype=tf.int16)
    mask_end_token_sum = tf.math.cumsum(mask_end_token,axis=-1) # following end will have value greater than 1
    mask_end_token_sum = tf.cast(tf.where(mask_end_token_sum==1,1,0),dtype=tf.bool) 
    mask_end_token = tf.cast(mask_end_token,dtype=tf.bool)

    mask_end_token = tf.math.logical_and(mask_end_token,mask_end_token_sum)

    mask_not_end = tf.cast(tf.math.logical_not(mask_end_token),dtype=tf.float32)
    mask_not_end = tf.cast(tf.math.cumprod(mask_not_end,axis=-1),tf.bool)
    mask = tf.math.logical_or(mask_not_end, mask_end_token)
    return mask