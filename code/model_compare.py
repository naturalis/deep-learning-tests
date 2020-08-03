import os, sys, json
# import tensorflow as tf
# import numpy as np
# from sklearn.metrics import classification_report
from lib import baseclass

class ModelCompare(baseclass.BaseClass):

    def __init__(self):
        super().__init__()

    def fuck(self):
        for entry in os.scandir(self.models_folder):
            analysis = os.path.join(self.models_folder, entry.name, "analysis.json")
            classes = os.path.join(self.models_folder, entry.name, "classes.json")
            dataset = os.path.join(self.models_folder, entry.name, "dataset.json")
            if os.path.exists(dataset):
                with open(dataset) as json_file:
                    dataset = json.load(json_file)
                    print("{} ({})".format(dataset["model_name"],dataset["created"]))
                    print("\"{}\"".format(dataset["model_note"]))
                    print("classes: {} (image min/max: {} / {})".format(dataset["class_count"],dataset["class_image_minimum"],dataset["class_image_maximum"]))
                    print("epoch: {}".format("; ".join(dataset["training_phases"]["epochs"])))
                    print("frozen layers: {}".format("; ".join(dataset["training_phases"]["freeze_layers"])))
        
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
    compare.fuck()

