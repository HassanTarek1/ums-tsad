import tensorflow as tf

class EstimationNet(tf.keras.Model):
    def __init__(self, hidden_layer_sizes, activation=tf.nn.relu):
        super(EstimationNet, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(units=size, activation=activation, name=f"layer_{i+1}")
                              for i, size in enumerate(hidden_layer_sizes[:-1])]
        self.dropout = tf.keras.layers.Dropout
        self.output_layer = tf.keras.layers.Dense(units=hidden_layer_sizes[-1], activation=None, name="logits")

    @tf.function
    def call(self, inputs, training=False, dropout_ratio=None):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
            if training and dropout_ratio is not None:
                x = self.dropout(rate=dropout_ratio)(x, training=training)
        logits = self.output_layer(x)
        return tf.nn.softmax(logits)
