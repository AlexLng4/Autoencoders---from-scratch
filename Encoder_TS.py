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

    def backward(self, x, grad_mu, grad_logvar):
        grad_h_mu = np.dot(self.W_mu.T, grad_mu)
        grad_h_logvar = np.dot(self.W_logvar.T, grad_logvar)
        grad_h = grad_h_mu + grad_h_logvar
        grad_h *= (1 - np.tanh(np.dot(self.W1, x) + self.b1) ** 2)

        # Gradients for weights and biases
        self.grad_W1 = np.dot(grad_h,x.T)
        self.grad_b1 = grad_h

        self.grad_W_mu = np.dot(grad_mu, np.tanh(np.dot(self.W1,x) + self.b1).T)
        self.grad_b_mu = grad_mu

        self.grad_W_logvar = np.dot( grad_logvar, np.tanh(np.dot(self.W1,x) + self.b1).T)
        self.gradb_logvar = grad_logvar

    def learn(self, learning_rate):

        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1
        self.W_mu -= learning_rate * self.grad_W_mu
        self.b_mu -= learning_rate * self.grad_b_mu
        self.W_logvar -= learning_rate * self.grad_W_logvar
        self.b_logvar -= learning_rate * self.b_logvar

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
        h = np.dot(self.W1, z) + self.b1
        # Derivative of tanh
        grand_tanh_for_h = (1- np.tanh(h) ** 2)

        # Gradients for weights and biases
        self.grad_W2 = np.dot(grad_recon, np.tanh(h).T)
        self.grad_b2 = grad_recon
        self.grad_W1 = np.dot((np.dot(self.W2.T, grad_recon) * grand_tanh_for_h), z.T)
        self.grad_b1 = np.dot(self.W2.T, grad_recon) * grand_tanh_for_h

    def learn(self, learning_rate):

        self.W1 -= learning_rate * self.grad_W1
        self.b1 -= learning_rate * self.grad_b1
        self.W2 -= learning_rate * self.grad_W2
        self.b2 -= learning_rate * self.grad_b2

