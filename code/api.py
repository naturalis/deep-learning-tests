# # -*- coding: utf-8 -*-
# """
# Created on Thu Jul 18 11:02:30 2019
# @author: Yan
# """
# # best ressources are https://scotch.io/bar-talk/processing-incoming-request-data-in-flask
# # https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html?source=post_page
# # https://towardsdatascience.com/deploying-a-keras-deep-learning-model-as-a-web-application-in-p-fc0f2354a7ff



import tensorflow as tf
from flask import Flask, jsonify
from flask_restful import Resource, Api, reqparse, request
from flask_jwt import JWT, jwt_required
from datetime import datetime
import logging, os, json, sys, uuid
import numpy as np


USERS = []
UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = {'jpg'}


app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('JWT_KEY')
app.config['PROPAGATE_EXCEPTIONS'] = True
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

logger = None
model_path = None
model_classes_path = None
model = None
classes = None

def set_model_path(path):
    global model_path
    model_path = path

def set_classes_path(path):
    global model_classes_path
    model_classes_path = path

def initialize(app):
    initialize_logger()
    # initialize_users()
    load_model()

def initialize_logger(log_level=logging.INFO):
    global logger

    if os.getenv('API_DEBUG')=="1":
        log_level=logging.DEBUG

    logger=logging.getLogger("API")
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(os.getenv('API_LOGFILE_PATH'))
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    if os.getenv('API_DEBUG')=="1":
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

def initialize_users():
    global USERS, logger

    for item in [ 'API_USER', 'API_PASS', 'API_USERID' ]:
        if (os.getenv(item)==None):
            logger.error("{} missing from ENV".format(item))
            set_service_available(False)

    if get_service_available() == True:
        USERS.append({
            "username" : os.getenv('API_USER'),
            "password" : os.getenv('API_PASS'),
            "userid" : os.getenv('API_USERID')
        })

def load_model():
    global model, model_path, classes, model_classes_path, logger
    model = tf.keras.models.load_model(model_path,compile=False)
    logger.info("loaded model {} ({})".format(model,model_path))
    with open(model_classes_path) as f:
        classes = json.load(f)
    logger.info("loaded {} classes ({})".format(len(classes),model_classes_path))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/",methods=["GET","POST"])
def root():
    return { "naturalis identify species by image api" : "v0.1" }


# @jwt_required()
@app.route("/identify",methods=["POST"])
def identify_image():
    global model, classes
    if request.method == 'POST':

        uploaded_files = request.files.getlist("image")

        if len(uploaded_files)<1:
            return { "error" : "no file" }
        else:
            file = uploaded_files[0]

        if file and allowed_file(file.filename):
            unique_filename = str(uuid.uuid4())
            unique_filename = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(unique_filename)

            x = tf.keras.preprocessing.image.load_img(unique_filename, target_size=(299,299))        
            x = tf.keras.preprocessing.image.img_to_array(x)
            x = np.expand_dims(x, axis=0)

            predictions = model.predict(x)
            predictions = predictions[0].tolist()
            classes = {k: v for k, v in sorted(classes.items(), key=lambda item: item[1])}
            predictions = dict(zip(classes.keys(), predictions))
            predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

            os.remove(unique_filename)

            return json.dumps(predictions)
    else:
        return { "error" : "method not allowed" }


# def log_usage(language="",key="",room="",hits=""):
#     global logger
#     endpoint=request.path
#     remote_addr=request.remote_addr
#     logger.info("{remote_addr} - {endpoint} - {params} - {hits}"
#         .format(remote_addr=remote_addr,endpoint=endpoint,params=json.dumps({"language":language,"key":key,"room":room}),hits=hits))


# def log_request_error(error="unknown error"):
#     global logger
#     endpoint=request.path
#     remote_addr=request.remote_addr
#     logger.error("{remote_addr} - {endpoint} - {error}".format(remote_addr=remote_addr,endpoint=endpoint,error=error))


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({ "error" : e.description }), 404


# jwt = JWT(app, verify, identity)

# @jwt.jwt_error_handler
# def customized_error_handler(e):
#     # print(e.error)
#     # print(e.description)
#     # print(e.status_code)
#     return jsonify({ "error" : e.error }), e.status_code


# class User(object):
#     def __init__(self, id):
#         self.id = id

#     def __str__(self):
#         return "User(id='%s')" % self.id

# def verify(username, password):
#     global USERS

#     if not (username and password):
#         return False

#     for index, user in enumerate(USERS):
#         if user['username'] == username and user['password'] == password:
#             return User(id=user['userid'])

# def identity(payload):
#     user_id = payload['identity']
#     return {"user_id": user_id}




if __name__ == '__main__':

    if len(sys.argv)>2:
        model_name = sys.argv[2]
    else:
        model_name = os.environ['API_MODEL_NAME']

    model_path = os.path.join(os.environ['PROJECT_ROOT'],"models",model_name)

    print(model_path)

    m = os.path.join(model_path,"model.hdf5")
    c = os.path.join(model_path,"classes.json")

    set_model_path(m)
    set_classes_path(c)

    initialize(app)

    app.run(debug=(os.getenv('API_FLASK_DEBUG')=="1"),host='0.0.0.0')

    # TODO: logging, tokens, users, gunicorn
    # curl -s  -XPOST  -F "image=@RMNH.AVES.1125_1.jpg"  http://0.0.0.0:5000/identify

    # .env
    # API_MODEL_NAME=20200804-142255
    # API_LOGFILE_PATH=/log/general.log
    # API_DEBUG=1
    # API_FLASK_DEBUG=0




