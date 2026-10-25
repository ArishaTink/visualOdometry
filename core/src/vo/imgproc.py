import numpy as np
from scipy.ndimage import gaussian_filter, sobel

def build_pyramid(img: np.ndarray, levels: int) -> list:
    pyramid = [img]
    for _ in range(levels - 1):
        smoothed = gaussian_filter(pyramid[-1], sigma=1.0)
        pyramid.append(smoothed[::2, ::2])
    return pyramid

def image_gradients(img: np.ndarray):
    Ix = sobel(img, axis=1) / 8.0
    Iy = sobel(img, axis=0) / 8.0
    return Ix, Iy