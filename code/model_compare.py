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
            print(entry.name)
            analysis = os.path.join(entry.name, "analysis.json")
            classes = os.path.join(entry.name, "classes.json")
            dataset = os.path.join(entry.name, "dataset.json")
            if os.path.exists(dataset):
                dataset = json.loads(dataset)
                print(dataset.model_name)
        
#         scan dirs
#         see if there is
#             classes
#             dataset
#             result

#         present
#             model name
#             note
#             created
#             class image max & min
#             num of classes
#             epochs
#             freeze layers

#     accuracy                           0.90      6920
#    macro avg       0.83      0.79      0.78      6920
# weighted avg       0.89      0.90      0.89      692        






if __name__ == "__main__":

    compare = ModelCompare()

    compare.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    compare.set_project(os.environ)
    compare.fuck()

