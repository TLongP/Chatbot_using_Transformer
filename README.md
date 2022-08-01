# Chatbot_using_Transformer

This is free to use, so do what ever you want.


## Need to do:
[ ] pretrained embedding for the transformer

## Beam Search

The Translator use a beam search for the best sentence. 

This translator works only with 1 sentence as input, since this will we will generate a batch size of the beam_width.

## metric
to use the loss and accuracy function we use a mask function for the prediction

![Alt text](pics/prediction_mask.png?raw=true "model")


We take the 'longer' mask between the prediction and true_sentence and only wwhere the mask is 1 this will count to the loss
