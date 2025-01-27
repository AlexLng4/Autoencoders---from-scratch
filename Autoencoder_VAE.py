import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch

from Encoder_TS import Encoder, Decoder


def generate_time_series(num_series, seq_length):
    t = np.linspace(0, 10, seq_length)  # Timpul
    x = np.sin(t) + 0.1 * np.random.randn(num_series, seq_length)  # Valori observate (sinus cu zgomot)
    return t, x

def plot_data():

    plt.figure(figsize=(12, 6))
    for i in range(5):
        plt.subplot(5, 1, i + 1)
        plt.plot(t, x[i], label=f"Seria {i + 1}")
        plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    def re_parameterize(mu, logvar):
        # calculate the variance for distribution
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std

    def reconstruction_loss(x, x_recon):
        return np.mean((x - x_recon) ** 2, axis=1)

if __name__ == '__main__':

    num_series = 1000  # Numărul de serii temporale
    seq_length = 100  # Lungimea fiecărei serii
    t, x = generate_time_series(num_series, seq_length)

    print("t shape:", t.shape)  # (100,)
    print("x shape:", x.shape)  # (1000, 100)
    plot_data()

    # Params
    input_dim = seq_length
    hidden_dim = 50
    latent_dim = 2
    learning_rate = 0.001
    epochs = 20
    batch_size = 32

    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)

