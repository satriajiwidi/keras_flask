import os
import time
import json


def get_model_detail(model_name):
    filepath = os.path.join(os.getcwd(),
                            'saved_models',
                            model_name+'.h5')
    created = time.ctime(os.path.getctime(filepath))

    images = get_model_graphs(model_name)

    model_detail = {
        'name': model_name,
        'created': created,
        'images': images
    }

    return model_detail


def get_model_params(model_name):
    filepath = os.path.join(os.getcwd(),
                            'model_params',
                            model_name+'.json')
    with open(filepath) as json_data:
        model_params = json.load(json_data)

    return model_params


def get_saved_models():
    filepath = os.path.join(os.getcwd(), 'saved_models')
    models = [file for _, _, files in os.walk(filepath)
              for file in files]

    return models


def save_model_params(model_params):
    save_dir = os.path.join(os.getcwd(), 'model_params')
    model_name = '{}.json'.format(model_params['model_name'])

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, model_name)

    with open(filepath, 'w') as outfile:
        json.dump(model_params, outfile)

    print('Model params saved.')


def save_model(model, model_name):
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name += '.h5'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


def get_model_graphs(model_name):
    filepath = os.path.join(os.getcwd(), 'graphs')
    images = []
    for _, _, files in os.walk(filepath):
        for file in files:
            tmp = '_'.join(file.split('_')[1:])
            tmp = tmp.split('.')[0]
            if model_name == tmp:
                images.append(file)

    return images
