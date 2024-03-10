import numpy as np
import tensorflow as tf

class GMM:
    """ Gaussian Mixture Model (GMM) for TensorFlow 2.x"""

    def __init__(self, n_comp):
        self.n_comp = n_comp
        # Initializations will happen in `create_variables`
        self.phi = self.mu = self.sigma = self.L = None
        self.training = False

    def create_variables(self, n_features):
        # Initializing variables without variable_scope, which is not needed in TensorFlow 2.x
        self.phi = tf.Variable(tf.zeros(shape=[self.n_comp]), dtype=tf.float32, name="phi")
        self.mu = tf.Variable(tf.zeros(shape=[self.n_comp, n_features]), dtype=tf.float32, name="mu")
        self.sigma = tf.Variable(tf.zeros(shape=[self.n_comp, n_features, n_features]), dtype=tf.float32, name="sigma")
        self.L = tf.Variable(tf.zeros(shape=[self.n_comp, n_features, n_features]), dtype=tf.float32, name="L")

    def fit(self, z, gamma):
        """Fit data to GMM model."""
        if self.phi is None:
            self.create_variables(z.shape[-1])

        gamma_sum = tf.reduce_sum(gamma, axis=0)
        self.phi.assign(tf.reduce_mean(gamma, axis=0))
        self.mu.assign(tf.einsum("ik,il->kl", gamma, z) / gamma_sum[:, None])

        z_centered = tf.sqrt(gamma[:, :, None]) * (z[:, None, :] - self.mu[None, :, :])
        self.sigma.assign(tf.einsum("ikl,ikm->klm", z_centered, z_centered) / gamma_sum[:, None, None])

        # For stability, add a small value to the diagonal
        min_vals = tf.linalg.diag(tf.ones(z.shape[-1], dtype=tf.float32)) * 1e-6
        self.L.assign(tf.linalg.cholesky(self.sigma + min_vals[None, :, :]))

        self.training = False

    def energy(self, z):
        """Calculate energy of each row of z."""
        if self.training and self.phi is None:
            self.create_variables(z.shape[1])

        z_centered = z[:, None, :] - self.mu[None, :, :]  # ikl
        v = tf.linalg.triangular_solve(self.L, tf.transpose(z_centered, [1, 2, 0]))  # kli

        log_det_sigma = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.L)), axis=1)

        d = z.shape[1]
        logits = tf.math.log(self.phi[:, None]) - 0.5 * (tf.reduce_sum(v**2, axis=1) + d * tf.math.log(2.0 * np.pi) + log_det_sigma[:, None])
        energies = -tf.reduce_logsumexp(logits, axis=0)

        return energies

    def cov_diag_loss(self):
        """Calculate loss based on the diagonal of the covariance matrix."""
        diag_loss = tf.reduce_sum(tf.math.divide(1, tf.linalg.diag_part(self.sigma)))

        return diag_loss
