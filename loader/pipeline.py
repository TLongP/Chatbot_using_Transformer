import tensorflow as tf



def processing_text(line):
    """
    preprocessing each line 
    remove special character . ? , !
    the data is split by tab
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




def add_start_and_end_tokens(input_data):
    """
    Add a start and end token to each sentence
    """
    data = tf.strings.join(["[START]",input_data,"[END]"],separator=" ")
    return data