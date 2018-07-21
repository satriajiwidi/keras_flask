from __future__ import print_function
from .utils import save_model
from .plot import make_plot, make_cm_plot

import numpy as np

from keras.callbacks import EarlyStopping
from keras.datasets import cifar10, mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam, RMSprop, SGD
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical

from keras_sequential_ascii import keras2ascii

np.random.seed(1000)

from keras import backend as K
print(K.tensorflow_backend._get_available_gpus())

from .img_utils import load_image, roll_one_pixel, \
    stack_all, sum_all, make_predict_result_img


def build_model(lr, batch_size, epochs, activation_function,
                optimizer, conv_depth, model_name):
    graph = K.get_session().graph

    with graph.as_default():
        adam = Adam(lr=lr, decay=1e-6)
        sgd = SGD(lr=lr, decay=1e-6)
        rmsprop = RMSprop(lr=lr, decay=1e-6)
        optimizers = {
            'adam': adam,
            'sgd': sgd,
            'rmsprop': rmsprop,
        }

        model = Sequential()

        filters = 32
        model.add(Conv2D(filters, kernel_size=(3, 3),
                         activation=activation_function,
                         input_shape=(32, 32, 3),
                         padding='same'))
        # model.add(Conv2D(filters, kernel_size=(3, 3),
        #                 activation=activation_function,
        #                 padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25, seed=123))

        filters *= 2

        for _ in range(conv_depth-1):
            model.add(Conv2D(filters, kernel_size=(3, 3),
                             activation=activation_function,
                             padding='same'))
            # model.add(Conv2D(filters, kernel_size=(3, 3),
            #                 activation=activation_function,
            #                 padding='same'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(rate=0.25, seed=123))
            filters *= 2

        model.add(Flatten())
        for _ in range(1):
            model.add(Dense(1024, activation=activation_function))
            model.add(Dropout(rate=0.5, seed=123))

        num_classes = 10
        model.add(Dense(num_classes, activation='softmax'))

        print(optimizers[optimizer])

        model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy'])

        keras2ascii(model)

        X_train, Y_train, X_test, Y_test = load_data(num_classes)

        history = model.fit(
            X_train, Y_train,
            batch_size=batch_size,
            shuffle=True,
            epochs=epochs,
            validation_data=(X_test, Y_test),
            verbose=1)

        save_model(model, model_name)

        make_plot(history, epochs, model_name)
        # make_cm_plot(model, X_test, Y_test)


def load_data(num_classes):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    # (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    # X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # print("X_train new shape", X_train.shape)
    # print("X_test new shape", X_test.shape)

    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train /= 255
    # X_test /= 255

    # Y_train = to_categorical(Y_train, num_classes)
    # Y_test = to_categorical(Y_test, num_classes)

    # return X_train, Y_train, X_test, Y_test
    return X_train[:1000], Y_train[:1000], \
        X_test[1000:1100], Y_test[1000:1100]


def make_prediction(model, img):
    graph = K.get_session().graph

    with graph.as_default():
        model = load_model(model)
        print(model, 'loaded!')

        print('Preprocessing images...')
        img = load_image()
        m, n = img.size[1], img.size[0]
        m_crop, n_crop = 32, 32
        img = roll_one_pixel(img, m_crop, n_crop)

        print('Begin prediction...')
        for img_i in img:
            tmp = img_i['im']
            tmp = tmp.astype('float32')
            tmp /= 255
            tmp = np.expand_dims(tmp, axis=0)
            pred = model.predict(tmp, batch_size=None,
                                 verbose=0, steps=None)
            pred = np.argmax(pred)
            tmp = np.zeros([m_crop, n_crop])
            tmp.fill(pred)
            img_i['im'] = tmp

        img = stack_all(img, m, n)
        img = sum_all(img, m, n)
        print('Prediction finished!')

        return make_predict_result_img(img, m, n)
