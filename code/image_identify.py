# -*- coding: utf-8 -*-

import sys, json
import tensorflow as tf
import numpy as np

class ImageIdentify:

    model = None

    def load_model(self,filepath):
        self.model = tf.keras.models.load_model(filepath,compile=False)

    def predict(self,image):
        x = tf.keras.preprocessing.image.load_img(image, target_size=(299,299))        
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = np.expand_dims(x, axis=0)
        prediction = self.model.predict(x)
        y_classes = prediction.argmax(axis=-1)
        print(y_classes)
        return json.dumps(prediction.tolist())

if __name__ == '__main__':

    image = sys.argv[1]

    predict = ImageIdentify()
    predict.load_model("/data/corvidae/models/20200319-120518.hdf5")
    x = predict.predict(image)
    print(x)
