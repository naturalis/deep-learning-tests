import os, sys, json, argparse, itertools, csv
import tensorflow as tf
import numpy as np
from lib import baseclass, utils

class ImageIdentify(baseclass.BaseClass):

    override_image_root_folder = None
    prepend_image_root_folder = None
    images = []
    results = []
    top = 0
    max_identifications = None
    identifications_done = 0
    identification_style = "original"
    identification_batch_size = 16
    batch_transformations = None
    output_file = None

    def set_image(self,image_path):
        self.images.append(image_path)
        self.logger.info("got {} images".format(len(self.images)));

    def set_images(self,image_paths):
        for item in image_paths.split(","):
            self.images.append(item)
        self.logger.info("got {} images".format(len(self.images)));

    def set_image_list(self,list_path):
        with open(list_path) as f:
            self.images = f.read().splitlines()
        self.logger.info("got {} images".format(len(self.images)));

    def set_image_csv_list(self,list_path,column,delimiter):
        with open(list_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=delimiter)
            for row in csv_reader:
                self.images.append(row[column])
        self.logger.info("got {} images".format(len(self.images)));

    def set_top(self,top):
        self.top = top

    def set_identification_style(self,identification_style):
        self.identification_style = identification_style

    def set_identification_batch_size(self,size):
        self.identification_batch_size = size

    def set_batch_transformations(self,transformations):
        self.batch_transformations = transformations

    def set_override_image_root_folder(self,folder):
        self.override_image_root_folder = folder

    def set_prepend_image_root_folder(self,folder):
        self.prepend_image_root_folder = folder

    def set_output_file(self,output_file):
        self.output_file = output_file

    def set_max_identifications(self,max_identifications):
        self.max_identifications = max_identifications

    def save_results(self,result):
        with open(self.output_file, "a") as myfile:
            myfile.write(json.dumps(result)+"\n")

    def predict_images(self):
        self.results = []
        for image in self.images:

            if self.override_image_root_folder:
                image = os.path.join(self.override_image_root_folder, os.path.basename(image))

            if self.prepend_image_root_folder:
                image = os.path.join(self.prepend_image_root_folder, image)

            if os.path.exists(image):
                predictions = self.predict_image(image)
                x = { "image" : image, "prediction" : predictions["predictions"] }
                if predictions['predictions_original']:
                    x["predictions_original"] = predictions["predictions_original"]

                self.save_results(x)
                # self.results.append(x)

                self.logger.info("{}: {} ({})".format(image,predictions["predictions"][0]["class"],predictions["predictions"][0]["prediction"]));

                self.identifications_done += 1

                if self.max_identifications and self.identifications_done >= self.max_identifications:
                    self.logger.warning("hit max identifications: {}".format(self.identifications_done));
                    break

            else:
                self.logger.warning("image doesn't exist: {}".format(image));

        # return json.dumps({ "project" : self.project_name, "model" : self.model_name, "predictions" : self.results })

    def predict_image(self,image):
        if self.override_image_root_folder:
            image = os.path.join(self.override_image_root_folder, os.path.basename(image))

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

        predictions_batch = None
        predictions_batch = None

        if self.identification_style in [ "original", "both" ]:
            predictions_original = self.model.predict(x)
            predictions_original = predictions_original[0].tolist()

        if self.identification_style in [ "batch", "both", "batch_incl_original" ]:
            batch = self.generate_augmented_image_batch(x)
            predictions_batch = self.model.predict_on_batch(batch)
            if self.identification_style == "batch_incl_original":
                predictions_original = predictions_batch[0].tolist()
            predictions_batch = np.mean(predictions_batch,axis=0)
            predictions_batch = predictions_batch.tolist()

        classes = {k: v for k, v in sorted(self.classes.items(), key=lambda item: item[1])}

        results_original = None
        results_batch = None

        if not predictions_original is None:
            predictions_original = dict(zip(classes.keys(), predictions_original))

            predictions_original = {k: v for k, v in sorted(predictions_original.items(), key=lambda item: item[1], reverse=True)}

            if self.top > 0:
                predictions_original = dict(itertools.islice(predictions_original.items(), self.top))

            results_original = []
            for key in predictions_original:
                results_original.append({ "class" : key, "prediction": predictions_original[key] })

        if not predictions_batch is None:
            predictions_batch = dict(zip(classes.keys(), predictions_batch))
            predictions_batch = {k: v for k, v in sorted(predictions_batch.items(), key=lambda item: item[1], reverse=True)}

            if self.top > 0:
                predictions_batch = dict(itertools.islice(predictions_batch.items(), self.top))

            results_batch = []
            for key in predictions_batch:
                results_batch.append({ "class" : key, "prediction": predictions_batch[key] })

        if not results_batch is None:
            output = { "predictions" : results_batch }
            if not results_original is None:
                output[ "predictions_original" ] = results_original
        else:
            output = { "predictions" : results_original }

        return output

    def generate_augmented_image_batch(self,original):
        b = self.batch_transformations

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            width_shift_range=b["width_shift_range"] if "width_shift_range" in b else [-0.1,-0.1],
            height_shift_range=b["height_shift_range"] if "height_shift_range" in b else [-0.1,-0.1],
            rotation_range=b["rotation_range"] if "rotation_range" in b else 5,
            zoom_range=b["zoom_range"] if "zoom_range" in b else 0.1
        )

        batch = []
        if self.identification_style == "batch_incl_original":
            batch.append(original[0])

        it = datagen.flow(original, batch_size=1)

        for i in range(self.identification_batch_size-len(batch)):
            next_batch = it.next()
            image = next_batch[0]
            batch.append(image)

        return np.array(batch)


if __name__ == '__main__':

    predict = ImageIdentify()
    predict.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    predict.set_project(os.environ)

    timer = utils.Timer()
    timer.get_time_passed()

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--images", type=str)
    parser.add_argument("--image_list", type=str)
    parser.add_argument("--image_csv_list", type=str)
    parser.add_argument("--csv_delimiter", type=str, default=",")
    parser.add_argument("--csv_column", type=int)
    parser.add_argument("--max_identifications", type=int)

    parser.add_argument("--model", type=str)
    parser.add_argument("--identification_style", choices=[ "original", "batch", "both", "batch_incl_original" ], default="batch_incl_original")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--output_file", type=str, default="./output.json")

    parser.add_argument("--override_image_root_folder",type=str)
    parser.add_argument("--prepend_image_root_folder",type=str)

    args = parser.parse_args()

    if args.model:
        predict.set_model_name(args.model)
    else:
        predict.set_model_name(os.environ['API_MODEL_NAME'])

    batch_size = os.getenv('API_BATCH_ID_SIZE')
    batch_transformations = os.getenv('API_BATCH_TRANSFORMATIONS')

    if args.identification_style:
        predict.set_identification_style(args.identification_style)

    if not batch_size is None:
        predict.set_identification_batch_size(int(batch_size))

    if not batch_transformations is None:
        predict.set_batch_transformations(json.loads(batch_transformations))


    # replaces whatever the folder is in the image list (just keeps basename)
    if args.override_image_root_folder:
        predict.set_override_image_root_folder(args.override_image_root_folder)

    # prepends to whatever is in the image list
    if args.prepend_image_root_folder:
        predict.set_prepend_image_root_folder(args.prepend_image_root_folder)

    predict.set_output_file(args.output_file)
    predict.set_model_folder()
    predict.load_model()
    predict.set_top(args.top)

    if args.max_identifications:
        predict.set_max_identifications(args.max_identifications)

    if args.image:
        # predict.set_image(args.image)
        json.dumps(predict.predict_image(args.image))
    elif args.images:
        predict.set_images(args.images)
        predict.predict_images()
    elif args.image_list:
        predict.set_image_list(args.image_list)
        predict.predict_images()
    elif args.image_csv_list:
        if not args.csv_column:
            raise ValueError("need column of images in CSV-file (--csv_column)")
        predict.set_image_csv_list(args.image_csv_list,args.csv_column,args.csv_delimiter)
        predict.predict_images()

    print(timer.get_time_passed(format="pretty"))
