import cv2
import keras.backend as K
import numpy as np
import os


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def load_dataset(image_dir):
    return os.listdir(image_dir)


def to_onehot(img, map):
    data = []
    for row in img:
        r = [map[tuple(pixel)] for pixel in row]
        data.append(r)
    return np.array(data, dtype=K.floatx())


def from_onehot(img):
    data = []
    converted = np.array([127, 127, 127])
    for row in img:
        r = [converted * np.argmax(pixel) for pixel in row]
        data.append(r)
    return np.array(data, dtype='uint8')


def load_image(path, image_size):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return cv2.resize(img[300:, :, :], image_size, interpolation=cv2.INTER_NEAREST)


def save_image(path, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def tversky_index(y_true, y_pred, alpha=0.5, beta=0.5):
    t, p = y_true[..., 1:], y_pred[..., 1:] # ignore background

    tp = K.sum(t * p) # true positives

    ones = K.ones_like(p)
    fp = K.sum(p * (ones - t))  # false positives
    fn = K.sum(t * (ones - p))  # false negatives

    T = tp / (tp + alpha * fp + beta * fn) # per class coeff

    return K.mean(T)
