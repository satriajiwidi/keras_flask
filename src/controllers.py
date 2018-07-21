import os
from flask import render_template, redirect, send_file, \
    url_for, request, flash
from .model import build_model, make_prediction
from .utils import get_model_detail, get_model_params, \
    save_model_params, get_saved_models, get_model_graphs
from .img_utils import check_image_to_predict


""" TEMPLATE RENDERING """


def index_page():
    return render_template('index.html')


def build_page():
    return render_template('pages/build.html')


def models_page():
    models = get_saved_models()

    return render_template('pages/models.html',
                           models=models)


def model_detail_page(model_name):
    model_detail = get_model_detail(model_name)
    model_params = get_model_params(model_name)

    return render_template('pages/model_detail.html',
                           model_detail=model_detail,
                           model_params=model_params)


def predict_page():
    filepath = os.path.join(os.getcwd(), 'saved_models')
    models = [file for _, _, files in os.walk(filepath)
              for file in files]

    return render_template('pages/predict.html',
                           models=models)


def result_page(model_name):
    images = get_model_graphs(model_name)

    return render_template('pages/result.html',
                           images=images,
                           model_name=model_name)


def result_predict_page():
    filename = 'result.png'
    is_exist = os.path.isfile(
        os.path.join(os.getcwd(), 'static/img/predict', filename))
    flag = 1
    if not is_exist:
        flag = 0

    return render_template('pages/result_predict.html',
                           filename=filename, flag=flag)


def predict_result_page(model, img):
    model = os.path.join(os.getcwd(), 'saved_models', model)
    img = make_prediction(model, img)

    return redirect(url_for('.result_predict_page'))


""" OTHER FUNCTIONALS """


def get_image(filename, flag):
    flag_allowed = [1, 2]
    if flag not in flag_allowed:
        return redirect('/')

    folder = 'graphs'
    if flag == 2:
        folder = 'static/img/predict'

    images = os.path.join(os.getcwd(), folder, filename)

    return send_file(images, mimetype='image/png')


def check_params(lr, batch_size, epochs, activation_function,
                 optimizer, conv_depth, model_name):
    lr_list = [0.01, 0.001, 0.0001]
    batch_size_list = [32, 64, 128, 256]
    epochs_list = [10, 30, 50, 80, 100]
    activation_function_list = ['relu', 'tanh', 'sigmoid']
    optimizer_list = ['adam', 'rmsprop', 'sgd']
    conv_depth_list = [1, 2, 3, 4]

    if lr not in lr_list or \
            batch_size not in batch_size_list or \
            epochs not in epochs_list or \
            activation_function not in activation_function_list or \
            optimizer not in optimizer_list or \
            conv_depth not in conv_depth_list or \
            not model_name:

        return None

    return lr, batch_size, epochs, activation_function, \
        optimizer, conv_depth, model_name


def build_submit(request):
    lr = float(request.form['lr'])
    batch_size = int(request.form['batch_size'])
    epochs = int(request.form['epochs'])
    activation_function = str(request.form['activation_function'])
    optimizer = str(request.form['optimizer'])
    conv_depth = int(request.form['conv_depth'])
    model_name = str(request.form['model_name'])

    result = check_params(
        lr, batch_size, epochs, activation_function,
        optimizer, conv_depth, model_name)

    if not result:
        return redirect('/build')

    lr, batch_size, epochs, activation_function, \
        optimizer, conv_depth, model_name = \
        result

    build_model(lr, batch_size, epochs, activation_function,
                optimizer, conv_depth, model_name)

    save_model_params({
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'activation_function': activation_function,
        'optimizer': optimizer,
        'conv_depth': conv_depth,
        'model_name': model_name
    })

    return redirect(url_for('.result_page',
                            model_name=model_name))


def upload(request, photos):
    filename = os.path.join(os.getcwd(),
                            'static/img/predict/predict.jpg')
    if os.path.exists(filename):
        os.remove(filename)

    if request.method == 'POST' and 'photo' in request.files:
        _ = photos.save(request.files['photo'],
                        name='predict.jpg')

        checks = check_image_to_predict()
        if checks is not None:
            flash(checks)
            return redirect('/predict')

        return predict_result_page(
            request.form['model'],
            filename)

    return redirect('/predict')
