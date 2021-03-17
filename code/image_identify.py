import os, sys, json, argparse
import tensorflow as tf
import numpy as np
from lib import baseclass, utils

class ImageIdentify(baseclass.BaseClass):

    images = []
    results = []
    top = 0

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

    def set_top(self,top):
        self.top = top


    def predict_images(self):
        self.results = []
        for image in self.images:
            if os.path.exists(image):
                predictions = self.predict_image(image)
                self.results.append({ "image" : image, "prediction" : predictions })
            else:
                print("image doesn't exist: {}".format(image));
        return json.dumps({ "project" : self.project_name, "model" : self.model_name, "predictions" : self.results })

    def predict_image(self,image):
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

        if identification_style in [ "original", "both" ]:
            predictions = model.predict(x)
            predictions = predictions[0].tolist()

            if identification_style == "both" :
                predictions_original = predictions

        if identification_style in [ "batch", "both", "batch_incl_original" ]:
            batch = generate_augmented_image_batch(x)
            batch_predictions = model.predict_on_batch(batch)
            predictions = np.mean(batch_predictions,axis=0)
            predictions = predictions.tolist()

            if identification_style  = "batch_incl_original":
                predictions_original = predictions[0].tolist()

        classes = {k: v for k, v in sorted(classes.items(), key=lambda item: item[1])}
        predictions = dict(zip(classes.keys(), predictions))
        predictions = {k: v for k, v in sorted(predictions.items(), key=lambda item: item[1], reverse=True)}

        if self.top > 0:
            count = 0
            topped = {}
            for k, v in predictions.items():
                topped[k]=v
                count += 1
                if count >= self.top:
                    break

            predictions = topped


        results = []
        for key in predictions:
            results.append({ 'class' : key, 'prediction': predictions[key] })

        if identification_style == "both" :
            predictions_original = dict(zip(classes.keys(), predictions_original))
            predictions_original = {k: v for k, v in sorted(predictions_original.items(), key=lambda item: item[1], reverse=True)}

            if self.top > 0:
                count = 0
                topped = {}
                for k, v in predictions_original.items():
                    topped[k]=v
                    count += 1
                    if count >= self.top:
                        break

                predictions_original = topped

            results_original = []
            for key in predictions_original:
                results_original.append({ 'class' : key, 'prediction': predictions_original[key] })



    def predict_image_original(self,image):
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

        if self.top > 0:
            count = 0
            topped = {}
            for k, v in predictions.items():
                topped[k]=v
                count += 1
                if count >= self.top:
                    break
            return topped
        else:
            return predictions



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
    predict.set_top(3)

    if args.image:
        # predict.set_image(args.image)
        x = json.dumps(predict.predict_image(args.image))

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