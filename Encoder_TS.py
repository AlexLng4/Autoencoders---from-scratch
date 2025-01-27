import numpy as np

class Encoder:

    def __init__(self, input_dim, hidden_dim, latent_dim):

        # normal neuron layer
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)

        # the neurons that produce the mean of the latent distribution
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_mu = np.zeros(hidden_dim)

        # the neurons that produce the log of the variance (square of the standard deviation)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)

    def forward(self, x):


        h = np.tanh(np.dot(x, self.W1) + self.b1)
        mu = np.dot(h, self.W_mu) + self.b_mu
        logvar = np.dot(h, self.W_logvar) + self.b_logvar

        return mu, logvar


class Decoder():

    def __init__(self, latent_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros(output_dim)

    def forward(self, z):

        h = np.tanh(np.dot(z, self.W1) + self.b1)
        x_recon = np.dot(h, self.W2) + self.W2
        return x_recon
