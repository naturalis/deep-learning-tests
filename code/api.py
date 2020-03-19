# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:02:30 2019
@author: Yan
"""
# best ressources are https://scotch.io/bar-talk/processing-incoming-request-data-in-flask
# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html?source=post_page
# https://towardsdatascience.com/deploying-a-keras-deep-learning-model-as-a-web-application-in-p-fc0f2354a7ff


import tensorflow as tf
from flask import Flask, request, jsonify

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'jpg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = None

def load_model(filepath):
    global model
    model = tf.keras.models.load_model(filepath)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            bla = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(bla)

            x = load_img(bla, target_size=(299,299))        
            x = img_to_array(x)
            x = np.expand_dims(x, axis=0)
            prediction = model.predict(x)
            return jsonify(prediction)
    else:
        return 'bla'
             


if __name__ == '__main__':
    load_model("/data/corvidae/models/")
    app.run(debug=False)