import os, sys, json, textwrap
# import tensorflow as tf
# import numpy as np
# from sklearn.metrics import classification_report
from lib import baseclass

class ModelCompare(baseclass.BaseClass):

    names = []
    dates = []
    notes = []
    classes = []
    epochs = []
    layers = []

    def __init__(self):
        super().__init__()

    def print_data(self):
        names = ""
        dates = ""
        notes = ""
        classes = ""
        epochs = ""
        layers = ""
        for x in range(len(self.names)):
            names += "** {:<23} ** "
            dates += "{:<30}"
            notes += "{:<30}"
            classes += "{:<30}"
            epochs += "{:<30}"
            layers += "{:<30}"
        print(names.format(*self.names))
        print(dates.format(*self.dates))
        # print(notes.format(*self.notes))
        print(classes.format(*self.classes))
        print(classes.format(*self.epochs))
        print(classes.format(*self.layers))

    def collect_data(self):
        self.names = []
        self.dates = []
        self.notes = []
        self.classes = []
        self.epochs = []
        self.layers = []

        for entry in os.scandir(self.models_folder):
            analysis = os.path.join(self.models_folder, entry.name, "analysis.json")
            classes = os.path.join(self.models_folder, entry.name, "classes.json")
            dataset = os.path.join(self.models_folder, entry.name, "dataset.json")
            if os.path.exists(dataset):
                with open(dataset) as json_file:
                    tmp = json.load(json_file)
                    self.names.append(tmp["model_name"])
                    self.dates.append(tmp["created"])
                    self.notes.append(textwrap.wrap(tmp["model_note"],30))
                    self.classes.append("{} ({}/{})".format(
                            tmp["class_count"],
                            tmp["class_image_minimum"],
                            tmp["class_image_maximum"]
                       )
                    )
                    self.epochs.append("{} {}".format("; ".join(map(str,dataset["training_phases"]["epochs"]))))
                    self.layers.append("{} {}".format("; ".join(map(str,dataset["training_phases"]["freeze_layers"]))))
                    
        
#         scan dirs
#         see if there is
#             classes
#             dataset
#             result


#     accuracy                           0.90      6920
#    macro avg       0.83      0.79      0.78      6920
# weighted avg       0.89      0.90      0.89      692        






if __name__ == "__main__":

    compare = ModelCompare()

    compare.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    compare.set_project(os.environ)
    compare.collect_data()
    compare.print_data()
