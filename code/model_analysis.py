import os, sys, json, argparse
import tensorflow as tf
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report
from lib import baseclass, dataset

class ModelAnalysis(baseclass.BaseClass):
    test_generator = None
    cm = None
    cp = None
    cm_exportable = []
    cp_exportable = None
    top_k = []
    class_tops=[]

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

        b = list(self.test_generator.class_indices.items())
        classes_1 = [v for k, v in b]
        classes_2 = [v for k, v in b]
        print(classes_1)
        print(classes_2)


    def do_analysis(self):
        batch_size = self.model_settings["batch_size"]
        Y_pred = self.model.predict(self.test_generator)
        y_pred = np.argmax(Y_pred, axis=1)

        #  confusion matrix
        self.cm = tf.math.confusion_matrix(self.test_generator.classes, y_pred)
        self.cm_exportable = []
        i = 0
        for row in self.cm:
            self.cm_exportable.append([])
            for cell in row:
                self.cm_exportable[i].append(str(cell.numpy()))
            i += 1

        # classification report
        self.cr = classification_report(self.test_generator.classes, y_pred)
        self.cr_exportable = classification_report(self.test_generator.classes, y_pred,output_dict=True)


        #  top k percentage overall
        for n in [1,3,5]:
            top_k = tf.math.in_top_k(self.test_generator.classes, Y_pred, n, "top_" + str(n))

            true_count = top_k.numpy()[np.where(top_k.numpy()==True)].size
            all_count = top_k.numpy().size

            self.top_k.append({"top" : n, "pct" : round((true_count / all_count) * 100,4) })

              
        #  top k count per class
        for class_key in np.unique(self.test_generator.classes):
            self.class_tops.append({"class" : int(class_key), "top_1" : 0, "top_3" : 0, "top_5" : 0 })

        for key, prediction in enumerate(Y_pred):
            this_class = [class_top for class_top in self.class_tops if class_top["class"] == self.test_generator.classes[key]]
            this_class = this_class[0]

            top1 = int(np.argmax(prediction))

            if self.test_generator.classes[key]==top1:
                this_class.update({"class" : self.test_generator.classes[key], "top_1" : (this_class["top_1"]+1)})

            if self.test_generator.classes[key] in prediction.argsort()[-3:][::-1]:
                this_class.update({"class" : self.test_generator.classes[key], "top_3" : (this_class["top_3"]+1)})

            if self.test_generator.classes[key] in prediction.argsort()[-5:][::-1]:
                this_class.update({"class" : self.test_generator.classes[key], "top_5" : (this_class["top_5"]+1)})

    def print_analysis(self):
        print("== confusion matrix ==")

        i = 0
        for row in self.cm:
            print("[ ",end='')
            for cell in row:
                print("{:>6d}".format(cell),end='')
            print(" ]")
        print("")

        print("== classification report ==")
        print(self.cr)
        print("")

        print("== top k analysis ==")
        print(self.top_k)
        print("")

        print("== top k count per class  ==")
        print(self.class_tops)
        print("")

    def backup_previous_analysis(self):
        if os.path.exists(self.get_analysis_path()):
            now = datetime.now()
            a = self.get_analysis_path()
            b = self.get_analysis_path(now.strftime('%Y%m%d%H%M%S'))
            os.rename(a,b)
            self.logger.info("moved existing analysis file {} to {}".format(a,b))

    def save_analysis(self):

        # for this_class in self.class_tops:
        #     this_class.update({
        #         "class" : this_class["class"].astype(np.int32)
        #         "top_1" : this_class["top_1"].astype(np.int32)
        #         "top_3" : this_class["top_3"].astype(np.int32)
        #         "top_5" : this_class["top_5"].astype(np.int32)
        #     })

        f = open(self.get_analysis_path(), "w")
        f.write(json.dumps({
            "confusion_matrix" : self.cm_exportable,
            "classification_report" : self.cr_exportable,
            "top_k" : self.top_k,
            "top_k_per_class" : self.class_tops
            }))
        f.close()

    def plot_confusion_matrix(self):
        df_cm = pd.DataFrame(self.cm, range(len(self.test_generator.classes)), range(len(self.test_generator.classes)))
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4) # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
        # plt.show()
        path = os.path.join(self.model_folder, "confusion_matrix.png")
        plt.savefig(path)
        self.logger.info("saved confusion matrix plot {}".format(path))



if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--load_model",type=str)
    args = parser.parse_args()

    analysis = ModelAnalysis()

    analysis.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    analysis.set_project(os.environ)

    if args.load_model: 
        analysis.set_model_name(args.load_model)
    else:
        raise ValueError("need a model name (--load_model=<model name>)")

    analysis.set_model_folder()

    dataset = dataset.DataSet()
    dataset.set_dataset_path(analysis.get_model_folder())
    dataset.open_dataset()

    analysis.load_model()
    analysis.set_class_image_minimum(dataset.get_dataset_value("class_image_minimum"))
    analysis.set_class_image_maximum(dataset.get_dataset_value("class_image_maximum"))

    analysis.read_class_list()
    analysis.read_image_list_file(image_col=2)
    analysis.image_list_apply_class_list()

    analysis.set_model_settings({
        "batch_size": dataset.get_dataset_value("training_settings")["batch_size"]
    })

    analysis.configure_generator()
    analysis.do_analysis()
    analysis.print_analysis()
    analysis.backup_previous_analysis()
    analysis.save_analysis()
    analysis.plot_confusion_matrix()
