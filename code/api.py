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
from time import perf_counter

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
identification_styles = [ "original", "batch", "both", "batch_incl_original" ]
identification_style = "original"
identification_batch_size = 16
batch_transformations = None

def set_model_path(path):
    global model_path
    model_path = path

def set_classes_path(path):
    global model_classes_path
    model_classes_path = path

def set_identification_style(style):
    global identification_style, identification_styles
    if style in identification_styles:
        identification_style = style
    else:
        raise ValueError("unknown identification style: {} ({})".format(style,";".join(identification_styles)))

def set_identification_batch_size(size):
    global identification_batch_size
    identification_batch_size = size

def set_batch_transformations(transformations):
    global batch_transformations
    batch_transformations = transformations

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
    global ALLOWED_EXTENSIONS
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/",methods=["GET","POST"])
def root():
    return { "naturalis identify species by image api" : "v0.1" }

# @jwt_required()
@app.route("/identify",methods=["POST"])
def identify_image():
    global logger, model, classes, identification_style

    if request.method == 'POST':

        uploaded_files = request.files.getlist("image")

        if len(uploaded_files)<1:
            return { "error" : "no file" }
        else:
            file = uploaded_files[0]

        logger.info("file: {}".format(file))

        if file and allowed_file(file.filename):

            unique_filename = str(uuid.uuid4())
            unique_filename = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(unique_filename)

            prediction_start = perf_counter()

            x = tf.keras.preprocessing.image.load_img(
                unique_filename,
                target_size=(299,299),
                interpolation="nearest")

            logger.info("identification_style: {}".format(identification_style))

            x = tf.keras.preprocessing.image.img_to_array(x)
            x = np.expand_dims(x, axis=0)

            x = x[..., :3]  # remove alpha channel if present
            if x.shape[3] == 1:
                x = np.repeat(x, axis=3, repeats=3)
            x /= 255.0
            # x = (x - 0.5) * 2.0 # why this, laurens?

            predictions_batch = None
            predictions_batch = None

            if identification_style in [ "original", "both" ]:
                predictions_original = model.predict(x)
                predictions_original = predictions_original[0].tolist()

            if identification_style in [ "batch", "both", "batch_incl_original" ]:
                batch = generate_augmented_image_batch(x)
                predictions_batch = model.predict_on_batch(batch)
                predictions_batch = np.mean(predictions_batch,axis=0)
                predictions_batch = predictions_batch.tolist()

                if identification_style == "batch_incl_original":
                    predictions_original = predictions_batch[0].tolist()

            os.remove(unique_filename)

            classes = {k: v for k, v in sorted(classes.items(), key=lambda item: item[1])}

            results_original = None
            results_batch = None

            if not predictions_original is None:
                predictions_original = dict(zip(classes.keys(), predictions_original))
                predictions_original = {k: v for k, v in sorted(predictions_original.items(), key=lambda item: item[1], reverse=True)}

                results_original = []
                for key in predictions_original:
                    results_original.append({ 'class' : key, 'prediction': predictions_original[key] })


            if not predictions_batch is None:
                predictions_batch = dict(zip(classes.keys(), predictions_batch))
                predictions_batch = {k: v for k, v in sorted(predictions_batch.items(), key=lambda item: item[1], reverse=True)}

                results_batch = []
                for key in predictions_batch:
                    results_batch.append({ 'class' : key, 'prediction': predictions_batch[key] })

            if not results_batch is None:
                logger.info("prediction (batch): {}".format(results_batch[0]))
            if not results_original is None:
                logger.info("prediction (original): {}".format(results_original[0]))

            logger.info("time taken: {}".format(perf_counter()-prediction_start))

            if not results_batch is None:
                output['predictions'] = results_batch
                if not results_original is None:
                    output['predictions_original'] = results_original
            else:
                output['predictions'] = results_original

            return json.dumps(output)

        else:
            return { "error" : "unsupported file type" }

    else:
        return { "error" : "method not allowed" }


def generate_augmented_image_batch(original):
    global logger, identification_style, identification_batch_size, batch_transformations

    logger.info("identification_batch_size: {}".format(batch_size))
    logger.info("batch_transformations: {}".format(batch_transformations))

    b = batch_transformations

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=b["width_shift_range"] if "width_shift_range" in b else [-0.1,-0.1],
        height_shift_range=b["height_shift_range"] if "height_shift_range" in b else [-0.1,-0.1],
        rotation_range=b["rotation_range"] if "rotation_range" in b else 5,
        zoom_range=b["zoom_range"] if "zoom_range" in b else 0.1
    )

    batch = []
    if identification_style == "batch_incl_original":
        batch.append(original[0])

    it = datagen.flow(original, batch_size=1)

    for i in range(identification_batch_size-len(batch)):
        next_batch = it.next()
        image = next_batch[0]
        batch.append(image)

    return np.array(batch)


@app.errorhandler(404)
def page_not_found(e):
    return jsonify({ "error" : e.description }), 404


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

    model_name = os.environ['API_MODEL_NAME']
    model_path = os.path.join(os.environ['PROJECT_ROOT'],"models",model_name)
    id_style = os.getenv('API_IDENTIFICATION_STYLE')
    batch_size = os.getenv('API_BATCH_ID_SIZE')
    batch_transformations = os.getenv('API_BATCH_TRANSFORMATIONS')

    m = os.path.join(model_path,"model.hdf5")
    c = os.path.join(model_path,"classes.json")

    set_model_path(m)
    set_classes_path(c)

    if not id_style is None:
        set_identification_style(id_style)

    if not batch_size is None:
        set_identification_batch_size(int(batch_size))

    if not batch_transformations is None:
        set_batch_transformations(json.loads(batch_transformations))

    initialize(app)

    app.run(debug=(os.getenv('API_FLASK_DEBUG')=="1"),host='0.0.0.0')

    # TODO: logging, tokens, users, gunicorn
    # curl -s -XPOST  -F "image=@ZMA.INS.1279115_1.jpg" http://0.0.0.0:8090/identify

    # .env
    # API_MODEL_NAME=20200804-142255
    # API_LOGFILE_PATH=/log/general.log
    # API_DEBUG=1
    # API_IDENTIFICATION_STYLE=batch (batch, original, both, batch_incl_original)
    # API_BATCH_ID_SIZE=16
    # API_FLASK_DEBUG=0 (avoid)


