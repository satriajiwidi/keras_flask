import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns


def make_plot(history, epochs, model_name):
    model_history = history.history

    save_dir = os.path.join(os.getcwd(), 'graphs')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # step = epochs/10

    plt.figure(0)
    plt.plot(model_history['acc'], 'r')
    plt.plot(model_history['val_acc'], 'g')
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['font.size'] = 10
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train: ' + str(round(model_history['acc'][-1], 2)),
                'validation: ' +
                str(round(model_history['val_acc'][-1], 2))])

    plt.savefig(os.path.join(save_dir,
                             'acc_{}.png'.format(model_name)))

    plt.figure(1)
    plt.plot(model_history['loss'], 'r')
    plt.plot(model_history['val_loss'], 'g')
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.rcParams['font.size'] = 10
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train: ' + str(round(model_history['loss'][-1], 2)),
                'validation:' +
                str(round(model_history['val_loss'][-1], 2))])

    plt.savefig(os.path.join(
        save_dir, 'loss_{}.png'.format(model_name)))


def make_cm_plot(model, X_test, Y_test):
    Y_pred = model.predict(X_test, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(np.argmax(Y_test, axis=1), y_pred)

    df_cm = pd.DataFrame(cm, range(10), range(10))
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='d')

    plt.savefig(os.path.join(os.getcwd(), 'graphs/cm.png'))
