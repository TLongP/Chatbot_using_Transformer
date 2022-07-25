import tensorflow as tf



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask_real = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask_pred = create_prediction_mask(pred)

    mask = tf.math.logical_or(mask_pred,mask_real)
    mask = tf.cast(mask,dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask_real = tf.math.logical_not(tf.math.equal(real, 0))


    mask_pred = create_prediction_mask(pred)
    mask = tf.math.logical_or(mask_pred,mask_real)

    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies,dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)


def create_prediction_mask(pred):
    """
    this will returns a mask with value 1 to the first end token
    """
    pred = tf.argmax(pred,axis=2)
    # returns the first end token in pred
    mask_end_token = tf.cast(tf.math.equal(pred,3),dtype=tf.int16)
    mask_end_token_sum = tf.math.cumsum(mask_end_token,axis=1) # followwing end wwill have value greater than 1
    mask_end_token_sum = tf.cast(tf.where(mask_end_token_sum==1,1,0),dtype=tf.bool) 
    mask_end_token = tf.cast(mask_end_token,dtype=tf.bool)

    mask_end_token = tf.math.logical_and(mask_end_token,mask_end_token_sum)

    mask_not_end = tf.cast(tf.math.logical_not(mask_end_token),dtype=tf.float32)
    mask_not_end = tf.cast(tf.math.cumprod(mask_not_end,axis=1),tf.bool)
    mask = tf.math.logical_or(mask_not_end, mask_end_token)
    return mask
