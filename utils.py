import numpy as np
from scipy.misc import imread, imresize

def get_noise(batch_size=32, num=100):
    return np.random.normal(0., 1., size=(batch_size, num))

