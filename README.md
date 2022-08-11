# Chatbot_using_Transformer

This is free to use, so do what ever you want.



# Beam Search

The Translator use a beam search for the best sentence. 

This Translator should also works with other architecture such as sequence to sequence model.

This translator works only with 1 sentence as input, since for each of top k sentences we want to focus to predict the next top k words for each of this sentence, we generate a batch size of k.

what you also can do is, to use tf.TensorArray to write the sentences tokens, this could boost the performance ( I did not test it).

## compute the loss for each sentence

Let $(y_1,...,y_n)$ be the first output sentence, $(w_1,...,w_m)$ be the second output sentence and $x$ be the given input sentence. We note that 

the probability to get the first sentence is

$$P(y,x) = P(y_1|x) * P(y_2|x,y_1) * ... * P(y_n|x,y_1,...,y_{n-1})$$

and the probability to get the second sentence is 

$$P(w,x) = P(w_1|x) * P(w_2|x,w_1) * ... * P(w_m|x,w_1,...,w_{m-1})$$

Note that the two sentence can have different length.

We search the sentence with the highest probability

$$ arg\ max \prod\limits_{i = 0}^{l_y} P(y_i|x,y_1,...,y_{i-1})$$

we can formulate this as the sum, since the product can get very small

$$ arg\ max \sum\limits_{i = 0}^{l_y} log(P(y_i|x,y_1,...,y_{i-1}))$$

$l_y$ denote the length of the sentence $y$.

```python
def _compute_sum_log_value(new_sentence,sentence_vec,end_token,alpha=0.7):
    """
    args:
    new_sentence : a tuple of token represents the predicted sentence
    sentence_vec : the value compute from the model from the new_sentence
    end_token : int, represents the end token for example '3' 
    

    compute 1/T^\alpha sum log(P(y_i|x,y_1,...,y_i-1))
    mask: is the true long of the sentence
    so that the sentence ends with the first end_token
    """
    mask = _create_prediction_mask(new_sentence,end_token)
    mask = tf.cast(mask,dtype=tf.float32)
    sentence_value = tf.math.log(tf.math.multiply(sentence_vec,mask))
    return tf.math.reduce_sum(sentence_value,axis=-1)/tf.math.pow(
                            tf.math.reduce_sum(mask,axis=-1),alpha) # normalize by length^alpha
```

## compute the mask for each sentence

In order to compute the probability for each sentence, we need to compute the mask for each sentence, so that this mask will have value 0 where the sentence ended.

```python
def _create_prediction_mask(pred,end_token):
    #first part
    mask_end_token = tf.cast(tf.math.equal(pred,end_token),dtype=tf.int16)
    mask_end_token_sum = tf.math.cumsum(mask_end_token,axis=-1) # following end will have value greater than 1
    mask_end_token_sum = tf.cast(tf.where(mask_end_token_sum==1,1,0),dtype=tf.bool) 
    mask_end_token = tf.cast(mask_end_token,dtype=tf.bool)
    # second part
    mask_end_token = tf.math.logical_and(mask_end_token,mask_end_token_sum)
    #third part
    mask_not_end = tf.cast(tf.math.logical_not(mask_end_token),dtype=tf.float32)
    mask_not_end = tf.cast(tf.math.cumprod(mask_not_end,axis=-1),tf.bool)
    mask = tf.math.logical_or(mask_not_end, mask_end_token)
    return mask
```
what the first part do is it create a mask for pred
mask_end_token_sum: end token will be mask with **$1$** and all following tokens will be mask with **$2$** or greater.
mask_end_token: will mask with 1 if it is an end token else 0

The second part will create a mask for the first end token all other will be 0.

The third part will create a mask for  pred, where the mask will be 1 until the first end token reach. All afterwards will be 0.

![Alt text](pics/prediction_mask.png?raw=true "model")







