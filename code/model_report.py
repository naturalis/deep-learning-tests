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

    def __init__(self):
        super().__init__()

    def read_classes(self):

        with open(self.get_classes_path()) as json_file:
            used_classes = json.load(json_file)

        with open(self.class_list_file_model) as csvfile:
            reader = csv.reader(csvfile)
            all_classes = list(filter(None, list(reader)))

        for key,val in used_classes.items():
            match = [ x for x in all_classes if 0 in x and x[0] == key ]
            print(match,key)
            self.classes.append({"key" : val, "class" : key, "support" : int(match[0][1]) })

        for item in all_classes:
            match = [ x for x in self.classes if x["class"] == item[0] ]
            if len(match)==0:
                self.skipped_classes.append({"class" : item[0], "support" : item[1] })


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

            for val in data["top_k_per_class"]:
                for d in self.classes:
                    if int(val["class"])==d["key"]:
                        d.update({
                            "top_1" : (val["top_1"] / d["support"]),
                            "top_3" : (val["top_3"] / d["support"]),
                            "top_5" : (val["top_5"] / d["support"])
                        })


    def print_report(self):
        l=0
        for item in self.classes:
            l = len(item["class"]) if len(item["class"]) > l else l

        s1 = "{: <"+str(l)+"}"
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

        print("-" * (l+65))

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

        print("")
        print("skipped classes")

        print(
            s1.format("class"),
            s2.format("support")
        )

        print("-" * (l+65))

        for item in self.classes:
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
    report.read_classes()
    report.read_analysis()
    report.print_report()
