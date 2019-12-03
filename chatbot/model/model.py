import tensorflow as tf


def scaled_dot_product_attention(query, key, value, mask):
    # calculate attention weights
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk 
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * 1e-9)

    # softmax is normalied on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, hyper_params, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = hyper_params.num_heads
        self.d_model = hyper_params.d_model

        assert self.d_model % self.num_heads == 0

        self.depth = self.d_model // self.num_heads 

        self.query_dense = tf.keras.layers.Dense(self.d_model)
        self.key_dense = tf.keras.layers.Dense(self.d_model)
        self.value_dense = tf.keras.layers.Dense(self.d_model)

        self.dense = tk.keras.layers.Dense(self.d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = input['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-product attention 
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention,
                (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_ln)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, hyper_params):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(hyper_params.vocab_size, 
                hyper_params.d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, 2( * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles 

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position = tf.trange(position, dtype=float32)[:, tf.newaxis],
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model = d_model)
        # apply sin to even indexes in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


def encoder_layer(hyper_params, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, hyper_params.d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(
            hyper_params = name="attention")({
                'query' : inputs,
                'key' : inputs, 
                'value' : inputs,
                'mask' : padding_mask
            })
    attention = tf.keras.layers.Dropout(hyper_params.dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + 
                                                                 attention)

    outputs = tf.keras.layers.Dense(
        hyper_params.num_units, activtion=hyper_params.activation)(attention)
    outputs = tf.keras.layers.Dense(hyper_params.d_model)(outputs)
    outputs = tf.keras.layers.Dropout(hyper_params.dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)


def encoder(hyper_params, name="encoder"):
    inputs = tf.keras.Input(shape=(None, ), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(hyper_params.vocab_size, 
                                            hyper_params.d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(hyper_params.d_model, tf.float32))
    embeddings = PositionalEncoding(hyper_params)(embeddings)

    outputs = tf.keras.layers.Dropout(hyper_params.dropout)(embeddings)

    for i in range(hyper_params.num_layers):
        outputs = encoder_layer(
            hyper_params,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

        return tf.keras.Model(
            inputs=[inputs, padding_mask], outputs=outputs, name=name)
