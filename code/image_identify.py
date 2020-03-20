# -*- coding: utf-8 -*-

import sys, json, os
import tensorflow as tf
import numpy as np

class ImageIdentify:

    model_path = None
    model_name = None
    model = None

    def set_model_path(self,path):
        self.model_path = path

    def set_model_name(self,name):
        self.model_name = name

    def load_model(self):
        self.model = tf.keras.models.load_model(os.path.join(self.model_path, self.model_name + ".hdf5"),compile=False)
        with open(os.path.join(self.model_path, self.model_name + "-classes.json")) as f:
            self.classes = json.load(f)

    def predict(self,image):
        x = tf.keras.preprocessing.image.load_img(image, target_size=(299,299))        
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = np.expand_dims(x, axis=0)

        predictions = self.model.predict(x)
        predictions = predictions[0].tolist()
        classes = {k: v for k, v in sorted(self.classes.items(), key=lambda item: item[1])}
        predictions = dict(zip(classes.keys(), predictions))
        predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

        return json.dumps(predictions)

if __name__ == '__main__':

    project_root = os.environ['PROJECT_ROOT']
    image = sys.argv[1]

    if 2 in sys.argv:
        model_name = sys.argv[2]
    else:
        model_name = os.environ['MODEL_NAME']

    predict = ImageIdentify()
    predict.set_model_path(os.path.join(project_root,"models"))
    predict.set_model_name(model_name)
    predict.load_model()
    x = predict.predict(image)
    print(x)

    # export PROJECT_ROOT=20200319-120518
    # export MODEL_NAME=20200319-120518
    # sudo docker-compose run tensorflow /code/image_identify.py /data/corvidae/images/eccbc87e4b/RMNH.AVES.47171_1.jpg 20200319-120518