import os, sys, json, textwrap, math
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
    accuracy_max = 0
    macro_precision_max = 0
    macro_recall_max = 0
    macro_f1_max = 0
    weighted_precision_max = 0
    weighted_recall_max = 0
    weighted_f1_max = 0

    def __init__(self):
        super().__init__()

    def print_data(self):

        print("")
        
        per_line = 2

        lines = math.ceil(len(self.names) / per_line)

        for i in range(lines):

            a = (i*per_line)
            b = (i*per_line)+per_line

            batch_names = self.names[a:b]
            batch_dates= self.dates[a:b]
            batch_model_sizes= self.model_sizes[a:b]
            batch_notes= self.notes[a:b]
            batch_classes= self.classes[a:b]
            batch_macro_support= self.macro_support[a:b]
            batch_epochs= self.epochs[a:b]
            batch_layers= self.layers[a:b]
            batch_accuracy = self.accuracy[a:b]
            batch_macro_precision = self.macro_precision[a:b]
            batch_weighted_precision = self.weighted_precision[a:b]
            batch_macro_recall = self.macro_recall[a:b]
            batch_weighted_recall = self.weighted_recall[a:b]
            batch_macro_f1 = self.macro_f1[a:b]
            batch_weighted_f1 = self.weighted_f1[a:b]

            general = ""

            for x in range(len(batch_names)):
                general += "{:<30}"

            index = "{:>12}"

            print(index.format("name: ") + general.format(*batch_names))
            print(index.format("date: ") + general.format(*batch_dates))

            notes = []
            max_l = 0
            for k,v in enumerate(batch_notes):
                t = textwrap.wrap(v.strip(),28,subsequent_indent="")
                notes.append(t)
                max_l = len(t) if len(t) > max_l else max_l

            for x in range(max_l):
                s = ""
                for note in notes:
                    try:
                        s += "{:<30}".format(note[x])
                    except IndexError:
                        s += "{:<30}".format("")
                        pass
                print(index.format("" if x > 0 else "note: ") + s)

            print(index.format("size: ") + general.format(*map(lambda x : x if x =="-" else str(math.ceil(x/1e6)) + "MB",batch_model_sizes)))
            print(index.format("classes: ") + general.format(*batch_classes))
            print(index.format("support: ") + general.format(*batch_macro_support))
            print(index.format("epochs: ") + general.format(*batch_epochs))
            print(index.format("frozen: ") + general.format(*batch_layers))
            print(index.format("accuracy: ") + general.format(*self._mark_max_val(self.accuracy_max,batch_accuracy)))
            print(index.format("precision: ") + general.format(*self._mark_max_val(self.macro_precision_max,batch_macro_precision)))
            print(index.format("") + general.format(*self._mark_max_val(self.weighted_precision_max,batch_weighted_precision)))
            print(index.format("recall: ") + general.format(*self._mark_max_val(self.macro_recall_max,batch_macro_recall)))
            print(index.format("") + general.format(*self._mark_max_val(self.weighted_recall_max,batch_weighted_recall)))
            print(index.format("f1: ") + general.format(*self._mark_max_val(self.macro_f1_max,batch_macro_f1)))
            print(index.format("") + general.format(*self._mark_max_val(self.weighted_f1_max,batch_weighted_f1)))

            print("")
            print("")


    def _mark_max_val(self,value_max,value_list):
        return map(lambda x : str(x) + " *" if x == value_max else x, value_list)

    def collect_data(self):
        self.names = []
        self.dates = []
        self.notes = []
        self.classes = []
        self.epochs = []
        self.layers = []
        self.model_sizes = []

        for entry in os.scandir(self.models_folder):
            analysis = os.path.join(self.models_folder, entry.name, "analysis.json")
            classes = os.path.join(self.models_folder, entry.name, "classes.json")
            dataset = os.path.join(self.models_folder, entry.name, "dataset.json")
            model = os.path.join(self.models_folder, entry.name, "model.hdf5")

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

                    self.accuracy_max = tmp["classification_report"]["accuracy"] \
                        if tmp["classification_report"]["accuracy"] > self.accuracy_max else self.accuracy_max
                    self.macro_precision_max = tmp["classification_report"]["macro avg"]["precision"] \
                        if tmp["classification_report"]["macro avg"]["precision"] > self.macro_precision_max else self.macro_precision_max
                    self.macro_recall_max = tmp["classification_report"]["macro avg"]["recall"] \
                        if tmp["classification_report"]["macro avg"]["recall"] > self.macro_recall_max else self.macro_recall_max
                    self.macro_f1_max = tmp["classification_report"]["macro avg"]["f1-score"] \
                        if tmp["classification_report"]["macro avg"]["f1-score"] > self.macro_f1_max else self.macro_f1_max
                    self.weighted_precision_max = tmp["classification_report"]["weighted avg"]["precision"] \
                        if tmp["classification_report"]["weighted avg"]["precision"] > self.weighted_precision_max else self.weighted_precision_max
                    self.weighted_recall_max = tmp["classification_report"]["weighted avg"]["recall"] \
                        if tmp["classification_report"]["weighted avg"]["recall"] > self.weighted_recall_max else self.weighted_recall_max
                    self.weighted_f1_max = tmp["classification_report"]["weighted avg"]["f1-score"] \
                        if tmp["classification_report"]["weighted avg"]["f1-score"] > self.weighted_f1_max else self.weighted_f1_max
            else:
                self.accuracy.append("")
                self.macro_precision.append("")
                self.macro_recall.append("")
                self.macro_f1.append("")
                self.macro_support.append("")
                self.weighted_precision.append("")
                self.weighted_recall.append("")
                self.weighted_f1.append("")
                self.weighted_support.append("")

            if os.path.exists(model):
                self.model_sizes.append(os.path.getsize(model))
            else:
                self.model_sizes.append("-")


if __name__ == "__main__":

    compare = ModelCompare()

    compare.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    compare.set_project(os.environ)
    compare.collect_data()
    compare.print_data()
