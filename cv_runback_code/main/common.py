import skimage.measure
import numpy as np
from PIL import Image
import os


def min_pooling(x):
    return skimage.measure.block_reduce(x, (3, 3), np.min)


def R_func(x):
    return 1 * (x > 0)


def G_func(x):
    return 1 * (x <= 0)


def B_func(x):
    mask0 = (x <= 0)
    mask1 = (x > 0)
    x[mask0] = 1
    x[mask1] = -1
    return x


# def capture():
#     with picamera.PiCamera() as camera:
#         camera.resolution = (640, 480)
#         time.sleep(1)
#         captured = camera.capture('captured.jpg')


def PILreshape(name):
    x = Image.open('{}.png'.format(name))
    x = x.resize(size=(256, 128))
    x.save('resize.png')


if __name__ == '__main__':
    os.chdir("G:\\raspi_cansat")
    PILreshape('image')