import os, json, csv, argparse
import numpy as np
from lib import baseclass

# import os, sys, json
# import tensorflow as tf
# import numpy as np
# from datetime import datetime
# from sklearn.metrics import classification_report
# from lib import baseclass, dataset

class ModelReport(baseclass.BaseClass):

    classes=[]
    skipped_classes=[]
    imperfect_classes = []
    max_class_name_length = 0
    max_skipped_class_name_length = 0
    this_model = {}

    def __init__(self):
        super().__init__()

    def read_dataset(self):
        with open(self.get_dataset_path()) as json_file:
            tmp = json.load(json_file)
            self.this_model["name"] = tmp["model_name"]
            self.this_model["date"] = tmp["created"]
            self.this_model["state"] = "?" if not "state" in tmp else tmp["state"]
            self.this_model["base_model"] = tmp["base_model"]
            self.this_model["training_time"] = tmp["training_time"] if "training_time" in tmp else "n/a"
            self.this_model["epochs_trained"] = tmp["epochs_trained"] if "epochs_trained" in tmp else "n/a"

            if "model_retrain_note" in tmp:
                self.this_model["note"] = "{} / {}".format(tmp["model_note"],tmp["model_retrain_note"])
            else:
                self.this_model["note"] = tmp["model_note"]

            self.this_model["class_count"] = int(tmp["class_count"])
            self.this_model["class_count_before_maximum"] = tmp["class_count_before_maximum"]
            self.this_model["class_image_minimum"] = tmp["class_image_minimum"]
            self.this_model["class_image_maximum"] = tmp["class_image_maximum"]
            self.this_model["use_class_weights"] = tmp["use_class_weights"]

            self.this_model["epochs"] = "; ".join(map(str,tmp["training_phases"]["epochs"]))
            self.this_model["layers"] = "; ".join(map(str,tmp["training_phases"]["freeze_layers"]))
            self.this_model["image_augmentation"] = tmp["training_settings"]["image_augmentation"].lower()!="none"
            self.this_model["downloaded_images_file"] = tmp["downloaded_images_file"] if "downloaded_images_file" in tmp else None
            self.this_model["class_list_file"] = tmp["class_list_file"] if "class_list_file" in tmp else None


    def read_classes(self):

        with open(self.get_classes_path()) as json_file:
            used_classes = json.load(json_file)

        with open(self.class_list_file_model) as csvfile:
            reader = csv.reader(csvfile)
            all_classes = list(filter(None, list(reader)))

        for key,val in used_classes.items():
            match = [ x for x in all_classes if x[0] == key ]
            self.classes.append({"key" : val, "class" : key, "support" : int(match[0][1]) })

        for item in all_classes:
            match = [ x for x in self.classes if x["class"] == item[0] ]
            if len(match)==0:
                self.skipped_classes.append({"class" : item[0], "support" : item[1] })

        # print(self.skipped_classes)


    def read_analysis(self):
        with open(self.get_analysis_path()) as json_file:
            data = json.load(json_file)
            for key,val in data["classification_report"].items():
                if (key.isnumeric()):
                    # print(key,val['precision'],val['recall'],val['f1-score'],val['support'])
                    for d in self.classes:
                        if int(key)==d["key"]:
                            # print(key,val['precision'],val['recall'],val['f1-score'],val['support'])
                            d.update({
                                "support" : int(val["support"]),
                                "precision" : float(val["precision"]),
                                "recall" : float(val["recall"]),
                                "f1-score" : float(val["f1-score"])}
                            )

                            if float(val["f1-score"]) < 1:
                                self.imperfect_classes.append({ 'key' : d["key"], 'f1-score' : val["f1-score"]})


            for val in data["top_k_per_class"]:
                for d in self.classes:
                    if int(val["class"])==d["key"]:
                        d.update({
                            "top_1" : (val["top_1"] / d["support"]),
                            "top_3" : (val["top_3"] / d["support"]),
                            "top_5" : (val["top_5"] / d["support"])
                        })


    def set_settings(self):
        self.max_class_name_length=0
        for item in self.classes:
            self.max_class_name_length = \
                len(item["class"]) if len(item["class"]) > self.max_class_name_length else self.max_class_name_length

        for item in self.skipped_classes:
            self.max_skipped_class_name_length = \
                len(item["class"]) if len(item["class"]) > self.max_skipped_class_name_length else self.max_skipped_class_name_length

    def print_report(self):
        self.set_settings(self)

        self.print_report_summary()
        print("")
        self.print_report_classes()
        print("")
        self.print_report_skipped_classes()

    def print_report_summary(self):

        s1 = "{: <13}"
        s2 = "{: >7}"

        print(s1.format("model ID:"),s2.format(self.this_model["name"]))
        print(s1.format("date:"),s2.format(self.this_model["date"]))
        print(s1.format("classes:"),s2.format(len(self.classes)))
        print(s1.format("classes:"),s2.format(self.this_model["class_count"]))
        print(s1.format("skipped:"),s2.format(len(self.skipped_classes)))
        print(s1.format("skipped:"),s2.format(self.this_model["class_count_before_maximum"] - self.this_model["class_count"]))
        print(s1.format("min. images:"),s2.format(self.this_model["class_image_maximum"]))
        print(s1.format("max. images:"),s2.format(self.this_model["class_image_maximum"]))



    def print_report_classes(self):
        s1 = "{: <"+str(self.max_class_name_length)+"}"
        s2 = "{: >7}"
        s3 = "{: >10}"
        s4 = "{: >10}"

        round_at = 5
        round_at_pct = 1

        print(
            s1.format("class"),
            s2.format("support"),
            s3.format("f1-score"),
            s3.format("precision"),
            s3.format("recall"),
            # s4.format("top_1"),
            s4.format("top 3"),
            s4.format("top 5"),
        )

        print("-" * (self.max_class_name_length+65))

        for item in self.classes:
            # print(item["class"],item["f1-score"],item["support"])
            print(
                s1.format(item["class"]),
                # s2.format(item["input"]),
                s2.format(item["support"]),
                s3.format(round(item["f1-score"],round_at)),
                s3.format(round(item["precision"],round_at)),
                s3.format(round(item["recall"],round_at)),
                # s4.format(round(item["top_1"],round_at_pct)), # top1 is the same as recall
                s4.format(round(item["top_3"],round_at)),
                s4.format(round(item["top_5"],round_at)),
            )

    def print_report_skipped_classes(self):
        s1 = "{: <"+str(self.max_skipped_class_name_length)+"}"
        s2 = "{: >7}"

        print(
            s1.format("skipped classes"),
            s2.format("support")
        )

        print("-" * (self.max_skipped_class_name_length+65))

        for item in self.skipped_classes:
            print(
                s1.format(item["class"]),
                s2.format(item["support"])
            )


    def save_report(self):
        pass
        # f = open(self.get_analysis_path(), "w")
        # f.write(json.dumps({
        #     "confusion_matrix" : self.cm_exportable,
        #     "classification_report" : self.cr_exportable,
        #     "top_k" : self.top_k,
        #     "top_k_per_class" : self.class_tops
        #     }))
        # f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_model",type=str)
    args = parser.parse_args()

    report = ModelReport()

    report.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    report.set_project(os.environ)

    if args.load_model:
        report.set_model_name(args.load_model)
    else:
        raise ValueError("need a model name (--load_model=<model name>)")

    report.set_model_folder()
    report.read_dataset()
    report.read_classes()
    report.read_analysis()
    report.print_report()
