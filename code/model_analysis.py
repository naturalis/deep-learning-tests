import os, sys, json, argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.metrics import classification_report
from lib import baseclass

class ModelAnalysis(baseclass.BaseClass):
    test_generator = None
    cm = None
    cp = None
    cm_exportable = []
    cp_exportable = None

    def __init__(self):
        super().__init__()

    def configure_generator(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
        )

        self.test_generator = datagen.flow_from_dataframe(
            dataframe=self.traindf,
            x_col=self.COL_IMAGE,
            y_col=self.COL_CLASS,
            class_mode="categorical",
            target_size=(299, 299),
            batch_size=self.model_settings["batch_size"],
            interpolation="nearest",
            shuffle=False
        )

    def do_analysis(self):
        batch_size = self.model_settings["batch_size"]
        Y_pred = self.model.predict(self.test_generator)
        y_pred = np.argmax(Y_pred, axis=1)

        self.cm = tf.math.confusion_matrix(self.test_generator.classes, y_pred)

        iets = tf.math.in_top_k(self.test_generator.classes, Y_pred, 3, "whatever")
        print(iets)

        self.cm_exportable = []
        i = 0
        for row in self.cm:
            self.cm_exportable.append([])
            for cell in row:
                self.cm_exportable[i].append(str(cell.numpy()))
            i += 1

        self.cr = classification_report(self.test_generator.classes, y_pred)
        self.cr_exportable = classification_report(self.test_generator.classes, y_pred,output_dict=True)

    def print_analysis(self):
        print("== confusion matrix ==")

        i = 0
        for row in self.cm:
            print("[ ",end='')
            for cell in row:
                print("{:>6d}".format(cell),end='')
            print(" ]")

        print("== classification report ==")
        print(self.cr)

    def backup_previous_analysis(self):
        if os.path.exists(self.get_analysis_path()):
            now = datetime.now()
            a = self.get_analysis_path()
            b = self.get_analysis_path(now.strftime('%Y%m%d%H%M%S'))
            os.rename(a,b)
            self.logger.info("moved existing analysis file {} to {}".format(a,b))

    def save_analysis(self):
        f = open(self.get_analysis_path(), "w")
        f.write(json.dumps({"confusion_matrix" : self.cm_exportable, "classification_report" : self.cr_exportable}))
        f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--load_model",type=str)
    args = parser.parse_args() 

    analysis = ModelAnalysis()

    analysis.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    analysis.set_project(os.environ)
    analysis.set_presets(os.environ)

    if args.load_model: 
        analysis.set_model_name(args.load_model)
    else:
        raise ValueError("need a model name (--load_model=<model name>)")

    analysis.set_model_folder()
    analysis.load_model()

    analysis.set_class_image_minimum(analysis.get_preset("class_image_minimum"))
    analysis.set_class_image_maximum(analysis.get_preset("class_image_maximum"))

    analysis.read_class_list()
    analysis.read_image_list_file(image_col=2)
    analysis.image_list_apply_class_list()

    analysis.set_model_settings({
        "batch_size": analysis.get_preset("batch_size")
    })

    analysis.configure_generator()
    analysis.do_analysis()
    analysis.print_analysis()
    analysis.backup_previous_analysis()
    analysis.save_analysis()
