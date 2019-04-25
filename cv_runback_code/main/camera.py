import numpy as np
import picamera
import time
from PIL import Image
import common


def camera_operation(num=-1):
    image = capture_image(num).transpose((2, 0, 1))  # (h, w, ch) -> (ch, h, w)
    only_red_image = to_only_red(image)
    target_x = get_RedCorn_coordinate(only_red_image)
    write_image_with_coordinate(only_red_image, target_x, num)

    w = only_red_image.shape[1]
    half_w = w // 2

    redzone = (only_red_image == 1)
    if only_red_image.size / 2 <= redzone.size:
        flag = 'ARRIVED'
    else:
        flag = 'yet'

    return (target_x, half_w, flag)


def capture_image(num):

    """
    :param num: for save image
    :return: numpy array image
    """

    with picamera.PiCamera() as camera:
        camera.resolution = (1024, 768)
        time.sleep(1)
        if num != -1:
            captured = camera.capture('captured_{}.jpg'.format(num))
        else:
            captured = camera.capture('captured_temp.jpg')

    return np.array(captured)


def to_only_red(image):

    """
    only red ch for detect red corn
    :param image: detect red corn image
    :return: image only red ch
    """

    # from image minus every ch median
    image_thresholds = [np.median(image[0]), np.median(image[1]), np.median(image[2])]
    normed_image = [image[i] - image_thresholds[i] for i in range(len(image_thresholds))]

    R = common.R_func(normed_image[0])       # R: 0 if x <= 0 else 1
    G = common.G_func(normed_image[1])       # G: 1 if x <= 0 else 0
    B = common.B_func(normed_image[2])       # B: 1 if x <= 0 else -1

    img = R * G * B
    img = common.R_func(img)
    img = common.min_pooling(img)

    return img


def get_RedCorn_coordinate(image):

    """
    赤コーンの存在する座標を特定する。
    :param image: 赤だけになった写真
    :return: 赤コーンがあると思われる座標。なかったときは”Nothing”と返す
    """

    col = np.sum(image, axis=0)       # (h, w) -> (w) 一次元に落とし込むと三角形の傾斜を検出しやすいと思った

    kernel = np.array([-1, 1])
    conv = np.convolve(col, kernel, mode='same')     # 一次元畳み込みで傾斜を求める

    col *= conv     # 赤コーンがあるところは傾斜が大きくて赤色も多いはず

    target_x = np.argmax(col)

    if col[target_x] != 0:
        return target_x
    else:
        return None


def write_image_with_coordinate(image, target_x, num):
    # imgの書き込み
    h, w = image.shape
    img = image.reshape(1, h, w) * 255
    g, b = np.zeros_like(img), np.zeros_like(img)
    img = np.concatenate((img, g, b)).transpose((1, 2, 0))

    # 目的地の座標を画像内で示す
    img[:, target_x, 1] = 255

    img = Image.fromarray(np.uint8(img)).resize((1024, 768))
    if num != -1:
        img.save('image_with_coordinate_{}.jpg'.format(num))
    else:
        img.save('image_with_coordinate_temp.jpg'.format(num))

