import os
import math

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from random import shuffle
from scipy.stats import mode


def load_image():
    filename = os.path.join(
        os.getcwd(), 'static/img/predict/predict.jpg')
    img = Image.open(filename)

    return img


def check_image_to_predict():
    img = load_image()
    img_format = str(img.format).lower()
    img_format_allowed = ['jpg', 'jpeg', 'png']
    # TODO:
    # update this max_area
    m = 300
    n = 60
    max_area = m*n
    img_area = img.size[0]*img.size[1]

    if img_format not in img_format_allowed:
        return 'Image format are not allowed. \
            Allowed formats are jpg, jpeg, or png'
    if img_area > max_area:
        return 'Max image area (width x height) is {}' \
            .format(max_area)

    return None


def roll_one_pixel(img, m_size, n_size):
    images = []
    m, n = list(reversed(img.size))
    for i in range(m-m_size+1):
        for j in range(n-n_size+1):
            new = img.crop((j, i, n_size+j, m_size+i))
            pix = np.array(new)
            images.append({
                'row_start': i,
                'row_end': m_size+i,
                'col_start': j,
                'col_end': n_size+j,
                'im': pix,
            })

    return images


def stack_all(images, m, n):
    print(len(images))
    a = np.zeros([m, n, len(images)])
    for i, img in enumerate(images):
        a[img['row_start']:img['row_end'],
          img['col_start']:img['col_end'], i] = img['im']

    return a


def sum_all(stacked_images, m, n):
    a = stacked_images
    c = np.zeros([m, n], dtype='int')
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            tmp = a[i, j, :]
            tmp = tmp[tmp.nonzero()]
            if tmp.size == 0:
                tmp = [0]
            tmp = mode(tmp).mode
            c[i, j] = tmp
    print(c.shape)

    return c


def make_predict_result_img(img, m, n):
    size = 20
    max_val = np.sort([m, n])[-1]
    m_size = math.ceil(m/max_val*size)
    n_size = math.ceil(n/max_val*size)
    n_size += 20/100*n_size
    plt.figure(figsize=(n_size, m_size))
    plot = sns.heatmap(img, fmt='d', cmap='summer_r')
    filename = os.path.join(os.getcwd(),
                            'static/img/predict/result.png')
    plot.get_figure().savefig(filename)

    return filename
