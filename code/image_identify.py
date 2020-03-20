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
        # sort classes
        classes = {k: v for k, v in sorted(self.classes.items(), key=lambda item: item[1])}
        predictions = dict(zip(classes.items(), predictions))
        return json.dumps(predictions)

if __name__ == '__main__':

    image = sys.argv[1]

    predict = ImageIdentify()
    predict.set_model_path("/data/corvidae/models")
    predict.set_model_name("20200319-120518")
    predict.load_model()
    x = predict.predict(image)
    print(x)

# sudo git pull; sudo docker-compose run tensorflow /code/image_identify.py /data/corvidae/images/eccbc87e4b/RMNH.AVES.47171_1.jpg