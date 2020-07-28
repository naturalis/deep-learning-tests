import os, sys, json
import tensorflow as tf
import numpy as np
from lib import baseclass

class ImageIdentify(baseclass.BaseClass):

    model_path = None
    model_name = None
    model = None

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

    predict = ImageIdentify()
    predict.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    predict.set_project(os.environ)

    image = sys.argv[1]

    if len(sys.argv)>2:
        model_name = sys.argv[2]
    else:
        model_name = os.environ['MODEL_NAME']

    predict.set_model_name(model_name)

    predict.load_model()

    x = predict.predict(image)
    print(x)

    # export PROJECT_ROOT=/data/ai/corvidae/
    # python image_identify.py /data/ai/corvidae/images/eccbc87e4b/RMNH.AVES.47171_1.jpg v1.0
    # sudo docker-compose run tensorflow /code/image_identify.py /data/corvidae/images/eccbc87e4b/RMNH.AVES.47171_1.jpg v1.0