import os, sys, json
import tensorflow as tf
import numpy as np
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

        self.cm_exportable = []
        i = 0
        for row in self.cm:
            self.cm_exportable.append([])
            for cell in row:
                self.cm_exportable[i].append(str(cell))
            i += 1

        self.cp = classification_report(self.test_generator.classes, y_pred)
        self.cp_exportable = classification_report(self.test_generator.classes, y_pred,output_dict=True)

    def print_analysis(self):
        print("== confusion matrix ==")

        i = 0
        for row in self.cm:
            print("[ ",end='')
            for cell in row:
                print("{:>6d}".format(cell),end='')
            print(" ]")

        print("== classification report ==")
        print(self.cp)

    def save_analysis(self):
        f = open(self.get_analysis_path(), "w")
        f.write(json.dumps({"confusion matrix" : self.cm_exportable, "classification report" : self.cp_exportable}))
        f.close()


if __name__ == "__main__":
    project_root = os.environ['PROJECT_ROOT']

    if len(sys.argv)>1:
        model_name = sys.argv[1]
    else:
        model_name = os.environ['MODEL_NAME']

    analysis = ModelAnalysis()

    analysis.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    analysis.set_project_folders(project_root=project_root)
    analysis.set_model_name(model_name)
    analysis.set_model_folder()

    analysis.load_model()

    analysis.read_image_list_file(image_col=2)
    analysis.set_class_list_file()
    analysis.read_class_list()

    analysis.set_model_settings({
        "batch_size": 64,
    })

    analysis.configure_generator()

    analysis.do_analysis()
    analysis.print_analysis()
    analysis.save_analysis()
