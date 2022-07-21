import tensorflow as tf



def processing_text(line):
    """
    preprocessing each line 
    remove special character . ? , !
    return is a 2d array  
    can also add to text vectorizer standardize
    """
    line = tf.strings.strip(line)
    line = tf.strings.regex_replace(line,"[\.\?\!,]", "")
    line = tf.strings.split(line,"\t")
    return line


def create_dataset(
                    path,
                    BUFFER_SIZE=10000,
                    BATCH_SIZE=64,
                    SEED=42
                    ):

    dataset = tf.data.TextLineDataset(path,buffer_size=BUFFER_SIZE)
    dataset = dataset.map(lambda x: processing_text(x))
    dataset = dataset.map(lambda x: {"question":x[0],"answer":x[1]})

    return dataset.cache(
                    ).shuffle(BUFFER_SIZE,seed=SEED
                    ).batch(BATCH_SIZE
                    ).prefetch(1)
 