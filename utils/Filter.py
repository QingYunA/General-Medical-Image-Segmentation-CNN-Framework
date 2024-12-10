import numpy as np
from scipy.ndimage import gaussian_filter, convolve

def low_pass_filter(data, cutoff, order):
    sigma = 1 / (np.pi * cutoff)
    return gaussian_filter(data, sigma=sigma)

def high_pass_filter(data, cutoff, order):
    low_passed = low_pass_filter(data, cutoff, order)
    return data - low_passed 