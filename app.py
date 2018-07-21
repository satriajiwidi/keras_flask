import os
from flask import Flask, request, redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES
import src.controllers as controllers


"""
DEFINE APP
"""
app = Flask(__name__)

app.secret_key = 'something-really-secret'

app.config["CACHE_TYPE"] = "null"

app.config.update(DEBUG=True,
                  TEMPLATES_AUTO_RELOAD=True)

# Photo upload config
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img/predict'
configure_uploads(app, photos)


"""
WITH CONTROLLERS
"""


@app.route('/')
def index_page():
    return controllers.index_page()


@app.route('/build')
def build_page():
    return controllers.build_page()


@app.route('/models')
def models_page():
    return controllers.models_page()


@app.route('/models/<string:model_name>')
def model_detail_page(model_name):
    return controllers.model_detail_page(model_name)


@app.route('/predict')
def predict_page():
    return controllers.predict_page()


@app.route('/result/<string:model_name>')
def result_page(model_name):
    return controllers.result_page(model_name)


@app.route('/result_predict')
def result_predict_page():
    return controllers.result_predict_page()


@app.route('/get-image/<int:flag>/<string:filename>')
def get_image(filename, flag):
    return controllers.get_image(filename, flag)


@app.route('/send-params', methods=['POST'])
def build_submit():
    return controllers.build_submit(request)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return controllers.upload(request, photos)


"""
MAIN EXCECUTION
"""
if __name__ == '__main__':
    app.run()
