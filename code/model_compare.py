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
    accuracy = []
    macro_precision = []
    macro_recall = []
    macro_f1 = []
    macro_support = []
    weighted_precision = []
    weighted_recall = []
    weighted_f1 = []
    weighted_support = []

    def __init__(self):
        super().__init__()

    def print_data(self):
        names = ""
        dates = ""
        notes = ""
        classes = ""
        epochs = ""
        layers = ""
        accuracy = ""
        for x in range(len(self.names)):
            names += "{:<30}"
            dates += "{:<30}"
            notes += "{:<30}"
            classes += "{:<30}"
            epochs += "{:<30}"
            layers += "{:<30}"
            accuracy += "{:<30}"

        index = "{:>10}"
        print(index.format("name:") + names.format(*map("*** {} ***".format,self.names)))
        print(index.format("date:") + dates.format(*self.dates))
        # print(index.format("note:") + notes.format(*self.notes))
        print(index.format("classes:") + classes.format(*self.classes))
        print(index.format("epochs:") + epochs.format(*self.epochs))
        print(index.format("frozen:") + layers.format(*self.layers))
        print(index.format("accuracy:") + accuracy.format(*self.accuracy))
        print(index.format("precision:") + accuracy.format(*self.macro_precision))
        print(index.format("(macro/weighed)") + accuracy.format(*self.weighted_precision))


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
                    self.notes.append(tmp["model_note"])
                    self.classes.append("{} ({}/{})".format(
                            tmp["class_count"],
                            tmp["class_image_minimum"],
                            tmp["class_image_maximum"]
                       )
                    )
                    self.epochs.append("; ".join(map(str,tmp["training_phases"]["epochs"])))
                    self.layers.append("; ".join(map(str,tmp["training_phases"]["freeze_layers"])))

            if os.path.exists(analysis):
                with open(analysis) as json_file:
                    tmp = json.load(json_file)
                    self.accuracy.append(tmp["classification_report"]["accuracy"])
                    self.macro_precision.append(tmp["classification_report"]["macro avg"]["precision"])
                    self.macro_recall.append(tmp["classification_report"]["macro avg"]["recall"])
                    self.macro_f1.append(tmp["classification_report"]["macro avg"]["f1-score"])
                    self.macro_support.append(tmp["classification_report"]["macro avg"]["support"])
                    self.weighted_precision.append(tmp["classification_report"]["weighted avg"]["precision"])
                    self.weighted_recall.append(tmp["classification_report"]["weighted avg"]["recall"])
                    self.weighted_f1.append(tmp["classification_report"]["weighted avg"]["f1-score"])
                    self.weighted_support.append(tmp["classification_report"]["weighted avg"]["support"])



if __name__ == "__main__":

    compare = ModelCompare()

    compare.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    compare.set_project(os.environ)
    compare.collect_data()
    compare.print_data()
