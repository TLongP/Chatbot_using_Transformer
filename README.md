# Chatbot_using_Transformer

This is free to use, so do what ever you want.


## Beam Search

The Translator use a beam search for the best sentence. 

This Translator should also works with other architecture such as sequence to sequence model.

This translator works only with 1 sentence as input, since for each of top k sentences we want to focus to predict the next top k words for each of this sentence, we generate a batch size of k.

what you also can do is, to use tf.TensorArray to write the sentences tokens, this could boost the performance ( I did not test it).

## metric
to use the loss and accuracy function we use a mask function for the prediction

![Alt text](pics/prediction_mask.png?raw=true "model")


We take the 'longer' mask between the prediction and true_sentence and only wwhere the mask is 1 this will count to the loss
