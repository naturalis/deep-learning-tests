import os, sys, json, textwrap, math, argparse
import shutil
from operator import itemgetter, attrgetter
from lib import baseclass

class ModelCompare(baseclass.BaseClass):

    cleanup = False
    delete = None
    sort = None
    reindex = False

    models = []
    broken_models = []

    accuracy_max = {}
    macro_precision_max = {}
    macro_recall_max = {}
    macro_f1_max = {}
    weighted_precision_max = {}
    weighted_recall_max = {}
    weighted_f1_max = {}

    def __init__(self):
        super().__init__()

    def set_cleanup(self,state):
        if isinstance(state,bool):
            self.cleanup = state

    def set_delete(self,models):
        self.delete = models.split(",")

    def set_sort(self,sort):
        self.sort = sort

    def clean_up(self):
        if self.cleanup:
            self.delete_broken()
        else:
            self.print_broken()

        if self.delete:
            self.delete_models()

        if self.reindex:
            self.collect_data()

    def delete_broken(self):
        if len(self.broken_models)==0:
            return

        print("deleting broken models (models w/o dataset.json or model.hdf5):")
        for item in self.broken_models:
            if input("{}: ".format("delete {} (y/n)?".format(item))).lower()=="y":
                shutil.rmtree(os.path.join(self.models_folder, item))
                print("deleted {}".format(item))
            else:
                print("skipped {}".format(item))

    def delete_models(self):
        if len(self.delete)==0:
            return

        print("deleting models:")
        for item in self.delete:
            if not os.path.exists(os.path.join(self.models_folder, item)):
                print("model doesn't exist: {}".format(item))
                continue

            if input("{}: ".format("delete {} (y/n)?".format(item))).lower()=="y":
                shutil.rmtree(os.path.join(self.models_folder, item))
                print("deleted {}".format(item))
                self.reindex = True
            else:
                print("skipped {}".format(item))

    def print_broken(self):
        if len(self.broken_models)==0:
            return

        print("broken models (no dataset.json):")
        for item in self.broken_models:
            print("{}".format(item))

    def superscript(self, n):
        return "".join(["⁰¹²³⁴⁵⁶⁷⁸⁹"[ord(c)-ord('0')] for c in str(n)])

    def _mark_max_val(self,value_max,value_list,variable):
        return map(lambda x : "{} * ({})".format(str(x[variable]),x["class_count"]) \
            if x[variable] == value_max[x["class_count"]] else x[variable], value_list)

    def _add_empty_values(self,this_model):
        this_model["accuracy"] = 0
        this_model["macro_precision"] = 0
        this_model["macro_recall"] = 0
        this_model["macro_f1"] = 0
        this_model["macro_support"] = 0
        this_model["weighted_precision"] = 0
        this_model["weighted_recall"] = 0
        this_model["weighted_f1"] = 0
        this_model["weighted_support"] = 0
        this_model["top_1"] = 0
        this_model["top_3"] = 0
        this_model["top_5"] = 0

        return this_model

    def collect_data(self):

        self.models = []

        folders = []

        for entry in os.scandir(self.models_folder):
            folders.append(entry.name)

        for folder in sorted(folders):

            self.set_model_name(folder)
            self.set_model_folder()

            analysis = self.get_analysis_path()
            classes = self.get_classes_path()
            dataset = self.get_dataset_path()
            model = self.get_model_path()

            if not os.path.exists(dataset) or not os.path.exists(model):
                self.broken_models.append(folder)
                continue

            this_model = {}

            with open(dataset) as json_file:
                tmp = json.load(json_file)
                this_model["name"] = tmp["model_name"]
                this_model["date"] = tmp["created"]
                this_model["state"] = "?" if not "state" in tmp else tmp["state"]
                this_model["base_model"] = tmp["base_model"]
                this_model["training_time"] = tmp["training_time"] if "training_time" in tmp else "n/a"
                this_model["epochs_trained"] = tmp["epochs_trained"] if "epochs_trained" in tmp else "n/a"

                if "model_retrain_note" in tmp:
                    this_model["note"] = "{} / {}".format(tmp["model_note"],tmp["model_retrain_note"])
                else:
                    this_model["note"] = tmp["model_note"]

                this_model["classes"] = "{} ({}) [{}…{}] {}".format(
                    tmp["class_count"],
                    "?" if not "class_count_before_maximum" in tmp else tmp["class_count_before_maximum"],
                    tmp["class_image_minimum"],
                    tmp["class_image_maximum"],
                    "?" if not "use_class_weights" in tmp else ("𐄷" if tmp["use_class_weights"] else "")
                    # "🗠"
                )

                this_model["class_count"] = int(tmp["class_count"])
                # this_model["use_class_weights"] = tmp["use_class_weights"]

                this_model["epochs"] = "; ".join(map(str,tmp["training_phases"]["epochs"]))
                this_model["layers"] = "; ".join(map(str,tmp["training_phases"]["freeze_layers"]))
                this_model["image_augmentation"] = tmp["training_settings"]["image_augmentation"].lower()!="none"
                this_model["downloaded_images_file"] = tmp["downloaded_images_file"] if "downloaded_images_file" in tmp else None
                this_model["class_list_file"] = tmp["class_list_file"] if "class_list_file" in tmp else None

                # oops
                this_model["downloaded_images_file"] = "?" if this_model["downloaded_images_file"] == None else os.path.basename(this_model["downloaded_images_file"])
                this_model["class_list_file"] =   "?" if this_model["class_list_file"] == None else os.path.basename(this_model["class_list_file"])


                if not this_model["class_count"] in self.accuracy_max:
                    self.accuracy_max[this_model["class_count"]] = 0

                if not this_model["class_count"] in self.macro_precision_max:
                    self.macro_precision_max[this_model["class_count"]] = 0

                if not this_model["class_count"] in self.macro_recall_max:
                    self.macro_recall_max[this_model["class_count"]] = 0

                if not this_model["class_count"] in self.macro_f1_max:
                    self.macro_f1_max[this_model["class_count"]] = 0

                if not this_model["class_count"] in self.weighted_precision_max:
                    self.weighted_precision_max[this_model["class_count"]] = 0

                if not this_model["class_count"] in self.weighted_recall_max:
                    self.weighted_recall_max[this_model["class_count"]] = 0

                if not this_model["class_count"] in self.weighted_f1_max:
                    self.weighted_f1_max[this_model["class_count"]] = 0


            if os.path.exists(model):
                this_model["model_size"] = os.path.getsize(model)
            else:
                this_model["model_size"] = "-"

            if os.path.exists(analysis):
                try:
                    with open(analysis) as json_file:
                        tmp = json.load(json_file)
                        this_model["accuracy"] = tmp["classification_report"]["accuracy"]
                        this_model["macro_precision"] = tmp["classification_report"]["macro avg"]["precision"]
                        this_model["macro_recall"] = tmp["classification_report"]["macro avg"]["recall"]
                        this_model["macro_f1"] = tmp["classification_report"]["macro avg"]["f1-score"]
                        this_model["macro_support"] = tmp["classification_report"]["macro avg"]["support"]
                        this_model["weighted_precision"] = tmp["classification_report"]["weighted avg"]["precision"]
                        this_model["weighted_recall"] = tmp["classification_report"]["weighted avg"]["recall"]
                        this_model["weighted_f1"] = tmp["classification_report"]["weighted avg"]["f1-score"]
                        this_model["weighted_support"] = tmp["classification_report"]["weighted avg"]["support"]
                        if "top_k" in tmp:
                            for item in tmp["top_k"]:
                                if item["top"]==1:
                                    this_model["top_1"] = float(item["pct"])
                                if item["top"]==3:
                                    this_model["top_3"] = float(item["pct"])
                                if item["top"]==5:
                                    this_model["top_5"] = float(item["pct"])
                        else:
                            this_model["top_1"] = 0
                            this_model["top_3"] = 0
                            this_model["top_5"] = 0

                        self.accuracy_max[this_model["class_count"]] = \
                            self.get_new_max(tmp["classification_report"]["accuracy"],
                                self.accuracy_max,
                                this_model["class_count"])

                        self.macro_precision_max[this_model["class_count"]] = \
                            self.get_new_max(tmp["classification_report"]["macro avg"]["precision"],
                                self.macro_precision_max,
                                this_model["class_count"])

                        self.macro_recall_max[this_model["class_count"]] = \
                            self.get_new_max(tmp["classification_report"]["macro avg"]["recall"],
                                self.macro_recall_max,
                                this_model["class_count"])

                        self.macro_f1_max[this_model["class_count"]] = \
                            self.get_new_max(tmp["classification_report"]["macro avg"]["f1-score"],
                                self.macro_f1_max,
                                this_model["class_count"])

                        self.weighted_precision_max[this_model["class_count"]] = \
                            self.get_new_max(tmp["classification_report"]["weighted avg"]["precision"],
                                self.weighted_precision_max,
                                this_model["class_count"])

                        self.weighted_recall_max[this_model["class_count"]] = \
                            self.get_new_max(tmp["classification_report"]["weighted avg"]["recall"],
                                self.weighted_recall_max,
                                this_model["class_count"])

                        self.weighted_f1_max[this_model["class_count"]] = \
                            self.get_new_max(tmp["classification_report"]["weighted avg"]["f1-score"],
                                self.weighted_f1_max,
                                this_model["class_count"])

                except ValueError as e:
                    print(e)
                    this_model = self._add_empty_values(this_model)
            else:
                this_model = self._add_empty_values(this_model)
                pass

            self.models.append(this_model)

    def get_new_max(self,new_value,value_list,class_count):
        return new_value \
            if new_value > value_list[class_count] \
            else value_list[class_count]

    def sort_data(self):
        if not self.sort is None:
            self.models = sorted(self.models, key=lambda row: (row[self.sort], row['class_count'],row['accuracy'],row['name']))
        else:
            self.models = sorted(self.models, key=lambda row: (row['class_count'],row['accuracy'],row['name']))


    def print_data(self):

        print("")

        per_line = 5
        max_col_width = 25

        lines = math.ceil(len(self.models) / per_line)

        for i in range(lines):

            a = (i*per_line)
            b = (i*per_line)+per_line

            batch_models = self.models[a:b]

            general = ""

            for x in range(len(batch_models)):
                general += "{:<30}"

            # index = "{:>12}"
            index = "{:>20}"

            print(index.format("name: ")  + general.format(*[x["name"] for x in batch_models]))
            print(index.format("date: ")  + general.format(*[x["date"][0:19] for x in batch_models]))
            print(index.format("state: ") +
                general.format(*[
                    "{} ({}/{})".format(x["state"],x["epochs_trained"],x["epochs"]) for x in batch_models]))
            print(index.format("base_model: ") + general.format(*[x["base_model"] for x in batch_models]))
            print(index.format("image_list: ") +
                general.format(*[
                    "{}…".format(x["downloaded_images_file"][:max_col_width])
                    if len(x["downloaded_images_file"]) > max_col_width
                    else x["downloaded_images_file"] for x in batch_models]))
            print(index.format("class_list: ") +
                general.format(*[
                    "{}…".format(x["class_list_file"][:max_col_width])
                    if len(x["class_list_file"]) > max_col_width
                    else x["class_list_file"] for x in batch_models]))

            notes = []
            max_l = 0
            for k,v in enumerate([x["note"] for x in batch_models]):
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

            # print(index.format("model_size: ") + \
            #     general.format(*map(lambda x : x if x =="-" else str(math.ceil(x/1e6)) + "MB",[x["model_size"] for x in batch_models])))

            print(index.format("classes: ") + \
                general.format(*[x["classes"] for x in batch_models]))

            print(index.format("support: ") + \
                general.format(*[x["macro_support"] for x in batch_models]))

            # print(index.format("epochs: ") + \
            #     general.format(*[x["epochs"] for x in batch_models]))

            print(index.format("frozen: ") + \
                general.format(*[x["layers"] for x in batch_models]))

            print(index.format("img aug: ") + \
                general.format(*[ "y" if x["image_augmentation"] else "n" for x in batch_models]))

            print(index.format("accuracy: ") + \
                general.format(*self._mark_max_val(self.accuracy_max,[x for x in batch_models],"accuracy")))

            print(index.format(" ├ top 1: ") + \
                general.format(*map(lambda x : str(round(x,2)) + "%",[x["top_1"] for x in batch_models])))

            print(index.format(" ├ top 3: ") + \
                general.format(*map(lambda x : str(round(x,2)) + "%",[x["top_3"] for x in batch_models])))

            print(index.format(" └ top 5: ") + \
                general.format(*map(lambda x : str(round(x,2)) + "%",[x["top_5"] for x in batch_models])))

            print(index.format("precision (macro): ") + \
                general.format(*self._mark_max_val(self.macro_precision_max,[x for x in batch_models],"macro_precision")))

            print(index.format("(weighted)  ") + \
                general.format(*self._mark_max_val(self.weighted_precision_max,[x for x in batch_models],"weighted_precision")))

            print(index.format("recall (macro): ") + \
                general.format(*self._mark_max_val(self.macro_recall_max,[x for x in batch_models],"macro_recall")))

            print(index.format("(weighted)  ") + \
                general.format(*self._mark_max_val(self.weighted_recall_max,[x for x in batch_models],"weighted_recall")))

            print(index.format("f1 (macro): ") + \
                general.format(*self._mark_max_val(self.macro_f1_max,[x for x in batch_models],"macro_f1")))

            print(index.format("(weighted)  ") + \
                general.format(*self._mark_max_val(self.weighted_f1_max,[x for x in batch_models],"weighted_f1")))

            print("")
            print("")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup", action='store_true')
    parser.add_argument("--delete", type=str, help='comma separated list of models to be deleted')
    parser.add_argument("--sort", type=str, help='add field to sort on (accuracy, base_model, model_size, date, etc.). default is: class_count,accuracy,name/date')
    args = parser.parse_args()

    compare = ModelCompare()

    compare.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    compare.set_project(os.environ)

    if args.cleanup:
        compare.set_cleanup(args.cleanup)

    if args.delete:
        compare.set_delete(args.delete)

    if args.sort:
        compare.set_sort(args.sort)

    compare.collect_data()
    compare.clean_up()
    compare.sort_data()
    compare.print_data()



