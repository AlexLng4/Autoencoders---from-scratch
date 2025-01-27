import numpy as np
from tensorflow.keras.datasets import  mnist
import matplotlib.pyplot as plt


def download_data():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Salvează datele local
    np.save('x_train.npy', x_train)
    np.save('y_train.npy', y_train)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)

    print("Datele au fost salvate local!")

def load_data():
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test = np.load('x_test.npy')
    y_test = np.load('y_test.npy')

    return  x_train, y_train, x_test, y_test

def plot_train_data():
    # Afișează primele 10 imagini din setul de antrenament
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()
