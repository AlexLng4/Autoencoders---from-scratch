from uu import decode

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch

from Encoder_TS import Encoder, Decoder


def generate_time_series(seq_length):
    t = np.linspace(0, seq_length, num=seq_length)  # Timpul
    x = np.sin(t) + 0.1 * np.random.randn(seq_length)  # Valori observate (sinus cu zgomot)
    return t, x

def plot_data(x, x_predict=None):

    plt.figure(figsize=(32, 24))
    plt.plot(t, x, label=f"Seria train")
    if x_predict is not None:
        plt.plot(t, x_predict, label=f"Seria predict")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def re_parameterize(mu, logvar):

    # calculate the variance for distribution
    # return the point created with the distribution parameters

    std = np.exp(0.5 * logvar)
    eps = np.random.randn(*mu.shape)
    return mu + eps * std

def reconstruction_loss(x, x_recon):
    return np.mean((x - x_recon) ** 2)

def kl_divergence(mu, logvar):
    return -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))


if __name__ == '__main__':

    seq_length = 1000 # Lungimea fiecărei serii
    t, x = generate_time_series(seq_length)

    print("t shape:", t.shape)  # (100,)
    print("x shape:", x.shape)  # (1000, 100)
    # plot_data()

    # Params
    input_dim = seq_length
    hidden_dim = 100
    latent_dim = 2
    learning_rate = 0.01
    epochs = 100

    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)

    # retrain from existing weights

    x_batch = x
    x_batch = x_batch.reshape(x_batch.shape[0], 1)
    #plot_data(x_batch)
    for epoch in range(epochs):

        # if epoch == 5000:
        #     learning_rate = 0.0001
        # forward
        mu, logvar = encoder.forward(x_batch)
        z = re_parameterize(mu, logvar)
        x_recon = decoder.forward(z)

        # plot_data(x_batch, x_recon)

        # compute loss
        recon_loss = reconstruction_loss(x_batch, x_recon)
        kl_loss = kl_divergence(mu, logvar)
        total_loss = np.mean(recon_loss + kl_loss)

        grad_recon = (x_recon - x_batch)
        decoder.backward(z, grad_recon)

        grad_mu = mu
        grad_logvar = (logvar - 1 / logvar)
        encoder.backward(x_batch,grad_mu, grad_logvar)

        encoder.learn(learning_rate)
        decoder.learn(learning_rate)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
        if epoch == epochs-1:
            plot_data(x_batch, x_recon)


    with open('model_weights_encoder.npy', 'wb') as f:
        np.save(f, encoder.W1)
        np.save(f, encoder.b1)
        np.save(f, encoder.W_mu)
        np.save(f, encoder.b_mu)
        np.save(f, encoder.W_logvar)
        np.save(f, encoder.b_logvar)
    with open('model_weights_decoder.npy', 'wb') as f:
        np.save(f, decoder.W1)
        np.save(f, decoder.b1)
        np.save(f, decoder.W2)
        np.save(f, decoder.b2)

    with open('train_data.npy', 'wb') as f:
        np.save(f, t)
        np.save(f, x)









