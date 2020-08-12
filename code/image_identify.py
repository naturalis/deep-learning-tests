import os, sys, json, argparse
import tensorflow as tf
import numpy as np
from lib import baseclass, utils

class ImageIdentify(baseclass.BaseClass):

    images = []
    results = []

    def set_image(self,image_path):
        self.images.append(image_path)
        # print(self.images)

    def set_images(self,image_paths):
        for item in image_paths.split(","):
            self.images.append(item)
        # print(self.images)

    def set_image_list(self,list_path):
        with open(list_path) as f:
            self.images = f.read().splitlines() 
        # print(self.images)

    def predict_images(self):
        self.results = []
        for image in self.images:
            if os.path.exists(image):
                predictions = self.predict_image(image)
                self.results.append({ "image" : image, "prediction" : predictions })
            else:
                print("image doesn't exist: {}".format(image));
        return json.dumps({ "project" : self.project_name, "model" : self.model_name, "predictions" : self.results })

    def predict_image(self,image,json_encode=False):
        x = tf.keras.preprocessing.image.load_img(
            image, 
            target_size=(299,299),
            interpolation="nearest")
        x = tf.keras.preprocessing.image.img_to_array(x)
        x = np.expand_dims(x, axis=0)

        x = x[..., :3]  # remove alpha channel if present
        if x.shape[3] == 1:
            x = np.repeat(x, axis=3, repeats=3)
        x /= 255.0
        # x = (x - 0.5) * 2.0 # why this, laurens?

        predictions = self.model.predict(x)
        predictions = predictions[0].tolist()
        classes = {k: v for k, v in sorted(self.classes.items(), key=lambda item: item[1])}
        predictions = dict(zip(classes.keys(), predictions))
        predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

        print(type(predictions))

        return json.dumps(predictions) if json_encode else predictions

if __name__ == '__main__':

    predict = ImageIdentify()
    predict.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    predict.set_project(os.environ)

    timer = utils.Timer()
    timer.get_time_passed()

    parser = argparse.ArgumentParser() 
    parser.add_argument("--image", type=str)
    parser.add_argument("--images", type=str)
    parser.add_argument("--list", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args() 

    if args.model:
        predict.set_model_name(args.model)
    else:
        predict.set_model_name(os.environ['API_MODEL_NAME'])

    predict.set_model_folder()
    predict.load_model()

    if args.image:
        # predict.set_image(args.image)
        x = predict.predict_image(args.image)

    if args.images:
        predict.set_images(args.images)
        x = predict.predict_images()

    if args.list:
        predict.set_image_list(args.list)
        x = predict.predict_images()

    print(timer.get_time_passed(format="pretty"))
    print(x)

    # export PROJECT_ROOT=/data/ai/corvidae/
    # python image_identify.py /data/ai/corvidae/images/eccbc87e4b/RMNH.AVES.47171_1.jpg v1.0
    # sudo docker-compose run tensorflow /code/image_identify.py /data/corvidae/images/eccbc87e4b/RMNH.AVES.47171_1.jpg v1.0
    # sudo docker-compose run tensorflow /code/image_identify.py --image=/data/corvidae/images/eccbc87e4b/RMNH.AVES.47171_1.jpg v1.0