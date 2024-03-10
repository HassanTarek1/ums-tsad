import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import time
import joblib
import logging
import torch as t

from .compression_net import CompressionNet, reconstruction_error
from .estimation_net import EstimationNet
from .gmm import GMM

from os import makedirs
from os.path import exists, join


class DAGMM(tf.keras.Model):
    """Deep Autoencoding Gaussian Mixture Model.

    This implementation is based on the paper:
    Bo Zong+ (2018) Deep Autoencoding Gaussian Mixture Model
    for Unsupervised Anomaly Detection, ICLR 2018
    (this is UNOFFICIAL implementation)
    """

    MODEL_FILENAME = "DAGMM_model"
    SCALER_FILENAME = "DAGMM_scaler"

    def __init__(
        self,
        comp_hiddens,
        est_hiddens,
        est_dropout_ratio=0.5,
        minibatch_size=1024,
        epoch_size=100,
        learning_rate=0.0001,
        lambda1=0.1,
        lambda2=0.0001,
        comp_activation=tf.nn.tanh,
        est_activation=tf.nn.tanh,
    ):
        """
        Parameters
        ----------
        comp_hiddens : list of int
            sizes of hidden layers of compression network
            For example, if the sizes are [n1, n2],
            structure of compression network is:
            input_size -> n1 -> n2 -> n1 -> input_sizes
        comp_activation : function
            activation function of compression network
        est_hiddens : list of int
            sizes of hidden layers of estimation network.
            The last element of this list is assigned as n_comp.
            For example, if the sizes are [n1, n2],
            structure of estimation network is:
            input_size -> n1 -> n2 (= n_comp)
        est_activation : function
            activation function of estimation network
        est_dropout_ratio : float (optional)
            dropout ratio of estimation network applied during training
            if 0 or None, dropout is not applied.
        minibatch_size: int (optional)
            mini batch size during training
        epoch_size : int (optional)
            epoch size during training
        learning_rate : float (optional)
            learning rate during training
        lambda1 : float (optional)
            a parameter of loss function (for energy term)
        lambda2 : float (optional)
            a parameter of loss function
            (for sum of diagonal elements of covariance)
        random_seed : int (optional)
            random seed used when fit() is called.
        """
        super(DAGMM, self).__init__()
        self.comp_net = CompressionNet(comp_hiddens, comp_activation)
        self.est_net = EstimationNet(est_hiddens, est_activation)
        self.est_dropout_ratio = est_dropout_ratio

        n_comp = est_hiddens[-1]
        self.gmm = GMM(n_comp)

        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.scaler = None

        self.graph = None
        self.sess = None

        self.time_tracker = {}

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    @tf.function
    def call(self, inputs, training=False):
        z, x_dash = self.comp_net(inputs, training=training)
        gamma = self.est_net(z, training=training)
        energy = self.gmm.energy(z)
        return x_dash, gamma, energy

    def train_step(self, x):
        with tf.GradientTape() as tape:
            x_dash, gamma, energy = self(x, training=True)
            loss = reconstruction_error(x, x_dash) + self.lambda1 * tf.reduce_mean(
                energy) + self.lambda2 * self.gmm.cov_diag_loss()
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


    def fit(self, train_dataloader):
        n_batches, n_features, n_time = train_dataloader.Y_windows.shape

        Y_windows = train_dataloader.Y_windows.reshape(n_batches * n_features * n_time, -1).reshape(-1, 1)

        self.fit2(x=Y_windows)


    @tf.function
    def fit2(self, x, epochs=100, batch_size=1024):
        """Fit the DAGMM model according to the given data.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Training data.
        """
        dataset = tf.data.Dataset.from_tensor_slices(x).batch(batch_size)
        for epoch in range(self.epoch_size):
            start = time.time()
            loss_metric = tf.keras.metrics.Mean()
            for batch in dataset:
                loss = self.train_step(batch)
                loss_metric(loss)
            avg_loss = loss_metric.result()
            print(f'Epoch {epoch + 1}, Loss: {avg_loss.numpy()}')
            loss_metric.reset_states()
            end = time.time()
            print(f'Time taken for epoch {epoch + 1}: {end - start} sec')


    def predict_prob(self, x):
        """Calculate anormaly scores (sample energy) on samples in X.

        Parameters
        ----------
        x : array-like, shape (n_samples, n_features)
            Data for which anomaly scores are calculated.
            n_features must be equal to n_features of the fitted data.

        Returns
        -------
        energies : array-like, shape (n_samples)
            Calculated sample energies.
        """
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        start = time.time()
        energies = self.sess.run(self.energy, feed_dict={self.input: x})
        end = time.time()
        self.time_tracker["test"] = end - start
        return energies

    def save(self, fdir):
        """Save trained model to designated directory.
        This method have to be called after training.
        (If not, throw an exception)

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
            If not exists, it is created automatically.
        """
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        if not exists(fdir):
            makedirs(fdir)

        model_path = join(fdir, self.MODEL_FILENAME)
        self.saver.save(self.sess, model_path)

    def restore(self, fdir):
        """Restore trained model from designated directory.

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
        """
        if not exists(fdir):
            raise Exception("Model directory does not exist.")

        model_path = join(fdir, self.MODEL_FILENAME)
        meta_path = model_path + ".meta"

        with tf.Graph().as_default() as graph:
            self.graph = graph
            config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(graph=graph, config=config)
            self.saver = tf.compat.v1.train.import_meta_graph(meta_path)
            self.saver.restore(self.sess, model_path)

            self.input, self.energy = tf.compat.v1.get_collection("save")

        if self.normalize:
            scaler_path = join(fdir, self.SCALER_FILENAME)
            self.scaler = joblib.load(scaler_path)


    def forward(self, input):
        Y = input['Y']
        n_batches, n_features, n_time = Y.shape

        t_Y = Y.reshape(n_batches * n_features * n_time).reshape(-1, 1).numpy()
        # print(f't_Y is {t_Y},shape is {t_Y.shape}')
        t_Y_score = self.predict_prob(t_Y)
        t_Y_score[np.isnan(t_Y_score)]=1.1
        # print(f'ori  t_Y_score is {t_Y_score}')
        t_Y_score = np.array([abs(i)for i in t_Y_score])
        t_Y_score[t_Y_score > 1.5] = 1.5
        t_Y_score = t_Y_score.reshape(-1, 1)
        # print(f't_Y_score is {t_Y_score}')
        Y_hat = t_Y * t_Y_score
        Y_hat = Y_hat.reshape(n_batches, n_features, n_time)

        return input['Y'], t.from_numpy(Y_hat), input['mask']
