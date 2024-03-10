import tensorflow as tf


def loss(x, x_dash):
    def euclid_norm(x):
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))

    norm_x = euclid_norm(x)
    norm_x_dash = euclid_norm(x_dash)
    dist_x = euclid_norm(x - x_dash)
    dot_x = tf.reduce_sum(x * x_dash, axis=1)

    min_val = 1e-3
    loss_E = dist_x / (norm_x + min_val)
    loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_dash + min_val))

    return tf.concat([loss_E[:, None], loss_C[:, None]], axis=1)


def reconstruction_error(x, x_dash):
    return tf.reduce_mean(tf.reduce_sum(tf.square(x - x_dash), axis=1))


class CompressionNet(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, activation=tf.nn.tanh):
        super(CompressionNet, self).__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.encoder_layers = [tf.keras.layers.Dense(units=size, activation=self.activation, name=f"encoder_layer_{n}")
                               for n, size in enumerate(hidden_layer_sizes[:-1])]
        self.encoder_output_layer = tf.keras.layers.Dense(units=hidden_layer_sizes[-1], activation=self.activation,
                                                          name="encoder_output_layer")
        self.decoder_layers = [tf.keras.layers.Dense(units=size, activation=self.activation, name=f"decoder_layer_{n}")
                               for n, size in enumerate(hidden_layer_sizes[:-1][::-1])]
        self.decoder_output_layer = tf.keras.layers.Dense(units=hidden_layer_sizes[0],
                                                          name="decoder_output_layer")  # Assuming input size matches the first hidden layer size

    def call(self, inputs, training=False):
        # Encoder
        z = inputs
        for layer in self.encoder_layers:
            z = layer(z)
        z_c = self.encoder_output_layer(z)

        # Decoder
        x_dash = z_c
        for layer in self.decoder_layers:
            x_dash = layer(x_dash)
        x_dash = self.decoder_output_layer(x_dash)

        # Compose feature vector (z_c concatenated with reconstruction loss features)
        z_r = loss(inputs, x_dash)
        z = tf.concat([z_c, z_r], axis=1)

        return z, x_dash
