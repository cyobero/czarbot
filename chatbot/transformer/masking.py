"""
create_padding_mask and create_look_ahead are helper functions
to creating masks to mask out padded tokens. We'll use these
helper functions as tf.keras.layers.Lambda layers.
"""

def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # (batch_size, 1, 1, sequence_length)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


