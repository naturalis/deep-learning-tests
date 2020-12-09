import os, json, argparse
from lib import baseclass, dataset

class ModelShow(baseclass.BaseClass):

    analysis = None
    classes = None
    dataset = None

    def __init__(self):
        super().__init__()

    def collect_data(self):
        # self.analysis = self.get_analysis_path()
        # self.classes = self.get_classes_path()
        dataset = self.get_dataset_path()
        with open(dataset) as json_file:
            self.dataset = json.load(json_file)

    def print_dataset(self):
        print(self.dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--load_model",type=str)
    args = parser.parse_args()

    show = ModelShow()

    show.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    show.set_project(os.environ)

    if args.load_model: 
        show.set_model_name(args.load_model)
    else:
        raise ValueError("need a model name (--load_model=<model name>)")

    show.set_model_folder()
    show.collect_data()
    show.print_dataset()



