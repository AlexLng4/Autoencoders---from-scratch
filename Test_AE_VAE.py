import numpy as np

if __name__=="__main__":



    with open('test.npy', 'rb') as f:
        a = np.load(f)
        b = np.load(f)
