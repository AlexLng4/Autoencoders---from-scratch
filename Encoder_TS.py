import numpy as np

class Encoder:

    def __init__(self, input_dim, hidden_dim, latent_dim):

        # normal neuron layer
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        # reshape to (hidden_dim, 1)
        self.b1 = self.b1.reshape(hidden_dim, 1)

        # the neurons that produce the mean of the latent distribution
        self.W_mu = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_mu = np.zeros(latent_dim)
        self.b_mu = self.b_mu.reshape(latent_dim, 1)


        # the neurons that produce the log of the variance (square of the standard deviation)
        self.W_logvar = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b_logvar = np.zeros(latent_dim)
        self.b_logvar = self.b_logvar.reshape(latent_dim, 1)

    def forward(self, x):

        h = np.tanh(np.dot(self.W1, x) + self.b1)
        mu = np.dot(self.W_mu, h) + self.b_mu
        logvar = np.dot(self.W_logvar, h) + self.b_logvar

        return mu, logvar


class Decoder():

    def __init__(self, latent_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.b1 = self.b1.reshape(hidden_dim, 1)

        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(output_dim)
        self.b2 = self.b2.reshape(output_dim, 1)

    def forward(self, z):

        h = np.tanh(np.dot(self.W1, z) + self.b1)
        x_recon = np.dot(self.W2, h) + self.b2
        return x_recon

    def backward(self, z, grad_recon):

        grad_recon = grad_recon.reshape(grad_recon.shape[0],1)
        grad_h = np.dot(grad_recon.T, self.W2.T)
        grad_h *= (1 - np.tanh(np.dot(z, self.W1) + self.b1) ** 2)  # Derivative of tanh

        # reshape
        z = z.reshape(z.shape[0], 1)

        # Gradients for weights and biases
        self.grad_W1 = np.dot(z, grad_h.T)
        self.grad_b1 = np.sum(grad_h, axis=0)
        self.grad_W2 = np.dot(np.tanh(np.dot(z.T, self.W1) + self.b1).T, grad_recon)
        self.grad_b2 = np.sum(grad_recon, axis=0)

