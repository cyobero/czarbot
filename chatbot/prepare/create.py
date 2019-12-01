"""
We are going to use the tf.data.Dataset API to contruct our input pipline in order to utilize
features like caching and prefetching to speed up the training process. The transformer is an
auto-regressive model: it makes predictions one part at a time, and uses its output so far to decide what to do next.

During training this example uses teacher-forcing. Teacher forcing is passing the true output 
to the next time step regardless of what the model predicts at the current time step.

As the transformer predicts each word, self-attention allows it to look at the previous words in the input sequence to better predict the next word.

To prevent the model from peaking at the expected output the model uses a look-ahead mask.

Target is divided into decoder_inputs which padded as an input to the decoder and cropped_targets for calculating our loss and accuracy.

"""
import tensorflow as tf

BATCH_SIZE = 64
BUFFER_SIZE = 35000

# decoder inputs use the previous target as input
# remove START_TOKEN from targets
dataset = tf.dataset.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
        },
    {
        'outputs': answers[:, :-1]
        },
    ))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
