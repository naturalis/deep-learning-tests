from __future__ import absolute_import, division, print_function, unicode_literals

import os, json, csv
import tensorflow as tf
import pandas as pd
import numpy as np
from shutil import copyfile
from datetime import datetime
from lib import logclass

def _determine_csv_separator(filepath,encoding):
    f = open(filepath, "r", encoding=encoding)
    line = f.readline()
    if line.count('\t') > 0:
        sep = '\t'
    else:
        sep = ','
    return sep

def _csv_to_dataframe(filepath, usecols, encoding="utf-8-sig"):
    return pd.read_csv(filepath, 
        encoding=encoding, 
        sep=_determine_csv_separator(filepath=filepath,encoding=encoding), 
        dtype="str", usecols=usecols, header=None)

class BaseClass():
    logger = None
    debug = False
    project_root = None
    project_name = None
    class_list_file_json = None
    class_list_file_csv = None
    image_list_file_csv = None
    downloaded_images_file = None
    timestamp = None
    model_name = None
    image_path = None
    models_folder = None
    image_list_class_col = None
    image_list_image_col = None
    class_list_file = None
    # class_list_file_class_col = None
    class_image_minimum = 2
    class_image_maximum = 0
    model_path = None
    architecture_path = None
    analysis_path = None
    traindf = None
    complete_class_list = []
    class_list = []
    model_settings = None
    COL_CLASS = "class"
    COL_IMAGE = "image"
    model = None
    presets = {}

    default_reduce_lr_params = { 'reduce_lr_params' : [ { "monitor" : "val_loss", "factor" : 0.1, "patience" : 4, "min_lr" : 1e-8, "verbose" : 1 } ] }


    def __init__(self):
        pr = os.environ.get("PROJECT_ROOT")

        if pr is None:
            raise ValueError("need a project root (PROJECT_ROOT missing from .env)") 

        if not os.path.exists(pr):
            os.mkdir(pr)

        logfile = os.path.join(pr, "log", "general.log")

        if not os.path.exists(logfile):
            path = os.path.join(pr, "log")
            if not os.path.exists(path):
                os.mkdir(path)
                os.chmod(path,0o777)

            with open(logfile, 'w') as fp: 
                pass
            os.chmod(logfile,0o777)

        self.logger = logclass.LogClass(self.__class__.__name__,logfile)
        self.logger.info("TensorFlow v{}".format(tf.__version__))
        self.set_timestamp()
        # if self.debug:
        #     tf.get_logger().setLevel("DEBUG")
        #     tf.autograph.set_verbosity(10)
        # else:
        #     tf.get_logger().setLevel("INFO")
        #     tf.autograph.set_verbosity(1)

    def set_timestamp(self):
        self.timestamp = datetime.now()

    def set_debug(self,state):
        self.debug = state

    def make_model_name(self):
        return "{0}{1:02d}{2:02d}-{3:02d}{4:02d}{5:02d}".format(
            self.timestamp.year,
            self.timestamp.month,
            self.timestamp.day,
            self.timestamp.hour,
            self.timestamp.minute,
            self.timestamp.second)

    def set_model_name(self,model_name=None):
        if not model_name is None:
            self.model_name = model_name
        else:
            raise ValueError("need a model name")

        self.logger.info("model name: {}".format(self.model_name))

    def set_project(self,os_environ):
        if 'PROJECT_NAME' in os_environ:
            self.set_project_name(os_environ.get("PROJECT_NAME"))
        else:
            raise ValueError("need a project name (PROJECT_NAME missing from .env)")

        if 'PROJECT_ROOT' in os_environ:
            self.set_project_folders(project_root=os_environ.get("PROJECT_ROOT"),image_path=os_environ.get("IMAGES_ROOT"))
        else:
            raise ValueError("need a project root (PROJECT_ROOT missing from .env)")

    def set_project_name(self, project_name):
        self.project_name = project_name
        self.logger.info("project name: {}".format(self.project_name))

    def set_project_folders(self, project_root, image_path=None):
        self.project_root = project_root

        if not os.path.isdir(self.project_root):
            raise FileNotFoundError("project root doesn't exist: {}".format(self.project_root))

        self.logger.info("project root: {}".format(self.project_root))

        self.class_list_file_json = os.path.join(self.project_root, "lists", "classes.json")
        self.class_list_file_csv = os.path.join(self.project_root, "lists", "classes.csv")
        self.image_list_file_csv = os.path.join(self.project_root, "lists", "images.csv")

        if image_path is not None:
            self.image_path = image_path
            self.logger.info("using override image folder {}".format(self.image_path))
        else:
            self.image_path = os.path.join(self.project_root, "images")

        if not os.path.exists(self.image_path):
            os.mkdir(self.image_path)
            self.logger.info("created image folder {}".format(self.image_path))

        self.models_folder = os.path.join(self.project_root, "models")

        if not os.path.exists(self.models_folder):
            os.mkdir(self.models_folder)
            self.logger.info("created models folder {}".format(self.models_folder))


        self.class_list_file = os.path.join(self.project_root, "lists", "classes.csv")
        self.downloaded_images_file = os.path.join(self.project_root, "lists", "downloaded_images.csv")

    def set_alternative_class_list_file(self,filename):
        alt_class_list_file = os.path.join(self.project_root, "lists", filename)
        if os.path.exists(alt_class_list_file):
            self.class_list_file = alt_class_list_file
        elif os.path.exists(filename):
            self.class_list_file = filename
        else:
            raise ValueError("alternative class list file {} does not exist".format(filename))

        self.logger.info("using alternative class list file {}".format(self.class_list_file))

    def set_model_folder(self):
        self.model_folder = os.path.join(self.project_root, "models", self.model_name)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
            self.logger.info("created model folder {}".format(self.model_folder))

        self.class_list_file_model = os.path.join(self.model_folder, "classes.csv")
        self.downloaded_images_file_model = os.path.join(self.model_folder, "downloaded_images.csv")

    def set_model_settings(self, model_settings):
        self.model_settings = model_settings
        for setting in self.model_settings:
            self.logger.info("setting - {}: {}".format(setting, str(self.model_settings[setting])))

        if "epochs" in self.model_settings:
            if isinstance(self.model_settings["epochs"], int):
                self.epochs = [self.model_settings["epochs"]]
            else:
                self.epochs = self.model_settings["epochs"]

    def set_class_image_minimum(self, class_image_minimum):
        class_image_minimum = int(class_image_minimum)
        if class_image_minimum >= 2:
            self.class_image_minimum = class_image_minimum
            self.logger.info("class minimum set to {}".format(self.class_image_minimum))
        else:
            raise ValueError("class minimum must be equal to or greater than 2 ({})".format(class_image_minimum))

    def set_class_image_maximum(self,class_image_maximum):
        self.class_image_maximum = int(class_image_maximum)
        self.logger.info("class maximum set to {}".format(self.class_image_maximum))

    def read_class_list(self):
        if not os.path.isfile(self.class_list_file_model):
            raise FileNotFoundError("class list file not found: {}".format(self.class_list_file_model))

        # self.class_list_file_class_col = class_col
        # self.class_list = _csv_to_dataframe(self.class_list_file_model, [self.class_list_file_class_col])
        tot_classes = 0        
        with open(self.class_list_file_model, 'r', encoding='utf-8-sig') as file:
            c = csv.reader(file)
            for row in c:
                tot_classes += 1
                self.complete_class_list.append(row)
                if int(row[1])>=self.class_image_minimum:
                    self.class_list.append(row)

        self.logger.info("retained {} classes (dropped {} due to image minimum of {})".format(
            len(self.class_list), str(tot_classes - len(self.class_list)), self.class_image_minimum))

    def get_class_list(self):
        return self.class_list

    def copy_class_list_file(self):
        copyfile(self.class_list_file,self.class_list_file_model)

    def copy_image_list_file(self):
        if not os.path.exists(self.downloaded_images_file_model):
            copyfile(self.downloaded_images_file, self.downloaded_images_file_model)
            self.logger.info("copied downloaded images file {}".format(self.downloaded_images_file_model))

    # TODO: implement Test split
    def read_image_list_file(self, class_col=0, image_col=1):
        self.image_list_class_col = class_col
        self.image_list_image_col = image_col

        self.logger.info("reading images from: {}".format(self.downloaded_images_file_model))

        skipped_images = 0

        if self.class_image_maximum > 0:

            self.logger.info("applying image maximum per class: {}".format(self.class_image_maximum))

            this_list=[]
            image_counter={}

            with open(self.downloaded_images_file_model) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=_determine_csv_separator(self.downloaded_images_file_model,"utf-8-sig"))
                for row in csv_reader:

                    this_class = row[self.image_list_class_col]

                    if this_class in image_counter and image_counter[this_class] >= self.class_image_maximum:
                        skipped_images += 1
                        continue

                    this_list.append([row[self.image_list_class_col].strip(),row[self.image_list_image_col].strip()])

                    if this_class in image_counter:
                        image_counter[this_class] += 1
                    else:
                        image_counter[this_class] = 1

            df = pd.DataFrame(this_list)

            self.logger.info("skipped {} images due to image maximum".format(skipped_images))

        else:

            df = _csv_to_dataframe(filepath=self.downloaded_images_file_model,
                                   usecols=[self.image_list_class_col, self.image_list_image_col])
        # if Test split
        #   df = df.sample(frac=1)
        #   msk = np.random.rand(len(df)) < 0.8
        #   self.traindf = df[msk]
        #   self.testdf = df[~msk]
        # # print(len(df), len(self.traindf), len(self.testdf))

        df.columns = [self.COL_CLASS, self.COL_IMAGE]
        df[self.COL_IMAGE] = self.image_path.rstrip("/") + "/" + df[self.COL_IMAGE].astype(str)
        self.traindf = df
        self.logger.info("read {} images in {} classes".format(self.traindf[self.COL_IMAGE].nunique(),self.traindf[self.COL_CLASS].nunique()))

    def image_list_apply_class_list(self):
        before = len(self.traindf)
        self.traindf = self.traindf[self.traindf[self.COL_CLASS].isin([i[0] for i in self.class_list if True])]
        after = len(self.traindf)
        self.logger.info("dropped {} out of {} images due to image minimum of {}".format(before - after, before, self.class_image_minimum))
        self.logger.info("retained {} images in {} classes".format(self.traindf[self.COL_IMAGE].nunique(),self.traindf[self.COL_CLASS].nunique()))

    def get_model_folder(self):
        return self.model_folder

    def get_model_path(self):
        self.model_path = os.path.join(self.model_folder, "model.hdf5")
        return self.model_path

    def get_architecture_path(self):
        self.architecture_path = os.path.join(self.model_folder, "architecture.json")
        return self.architecture_path

    def get_analysis_path(self,add_on=None):
        self.analysis_path = os.path.join(self.model_folder, "analysis{}.json".format("" if add_on is None else "--"+add_on))
        return self.analysis_path

    def get_classes_path(self):
        self.classes_save_path = os.path.join(self.model_folder, "classes.json")
        return self.classes_save_path

    def get_class_list_path(self):
        self.class_list_file_csv = os.path.join(self.project_root, "lists", "classes.csv")
        return self.class_list_file_csv

    def get_dataset_path(self):
        self.dataset_path = os.path.join(self.model_folder, "dataset.json")
        return self.dataset_path

    def load_model(self):
        self.model = tf.keras.models.load_model(self.get_model_path(),compile=False)
        self.logger.info("loaded model {}".format(self.get_model_path()))

        with open(os.path.join(self.get_classes_path())) as f:
            self.classes = json.load(f)

        self.logger.info("loaded {} classes from {}".format(len(self.classes),self.get_classes_path()))

    def set_presets(self,**kwargs):

        self.presets.update( { "use_tensorboard" : True } )
        self.presets.update( { "use_imagenet_weights" : True } )
        self.presets.update( { "use_class_weights" : False } )
        self.presets.update( { "early_stopping_monitor" : [ "val_loss" ] } )
        self.presets.update( { "early_stopping_patience" : [ 10 ] } )
        self.presets.update( { "image_list_class_column" : 0 } )
        self.presets.update( { "image_list_file_column" : 2 } )
        
        if 'os_environ' in kwargs:
            os_environ = kwargs['os_environ']

            base_model = os_environ.get("BASE_MODEL") if "BASE_MODEL" in os_environ else "InceptionV3"

            if 'IMAGE_AUGMENTATION' in os_environ:
                image_augmentation = json.loads(os_environ.get("IMAGE_AUGMENTATION"))
            else:
                image_augmentation = None

            if 'REDUCE_LR_PARAMS' in os_environ:
                reduce_lr_params = json.loads(os_environ.get("REDUCE_LR_PARAMS"))
            else:
                reduce_lr_params = self.default_reduce_lr_params

            validation_split = float(os_environ.get("VALIDATION_SPLIT")) if "VALIDATION_SPLIT" in os_environ else 0.2
            learning_rate = json.loads(os_environ.get("LEARNING_RATE")) if "LEARNING_RATE" in os_environ else [ 1e-4 ]
            batch_size = int(os_environ.get("BATCH_SIZE")) if "BATCH_SIZE" in os_environ else 64
            epochs = json.loads(os_environ.get("EPOCHS")) if "EPOCHS" in os_environ else [ 200 ]  
            freeze_layers = json.loads(os_environ.get("FREEZE_LAYERS")) if "FREEZE_LAYERS" in os_environ else [ "none" ]
            metrics = json.loads(os_environ.get("METRICS")) if "METRICS" in os_environ else [ "acc" ]
            checkpoint_monitor = os_environ.get("CHECKPOINT_MONITOR") if "CHECKPOINT_MONITOR" in os_environ else "val_acc"
            early_stopping_monitor = json.loads(os_environ.get("EARLY_STOPPING_MONITOR")) if "EARLY_STOPPING_MONITOR" in os_environ else [ "none" ]
            early_stopping_patience = json.loads(os_environ.get("EARLY_STOPPING_PATIENCE")) if "EARLY_STOPPING_PATIENCE" in os_environ else [ 10 ]
            class_image_minimum = int(os_environ.get("CLASS_IMAGE_MINIMUM")) if "CLASS_IMAGE_MINIMUM" in os_environ else 2
            class_image_maximum = int(os_environ.get("CLASS_IMAGE_MAXIMUM")) if "CLASS_IMAGE_MAXIMUM" in os_environ else 0
            use_tensorboard = (os_environ.get("USE_TENSORBOARD").lower()=="true") if "USE_TENSORBOARD" in os_environ else True
            use_imagenet_weights = (os_environ.get("USE_IMAGENET_WEIGHTS").lower()=="true") if "USE_IMAGENET_WEIGHTS" in os_environ else True
            use_class_weights = (os_environ.get("USE_CLASS_WEIGHTS").lower()=="true") if "USE_CLASS_WEIGHTS" in os_environ else False
            image_list_class_column = int(os_environ.get("IMAGE_LIST_CLASS_COLUMN")) if "IMAGE_LIST_CLASS_COLUMN" in os_environ else 0
            image_list_file_column = int(os_environ.get("IMAGE_LIST_FILE_COLUMN")) if "IMAGE_LIST_FILE_COLUMN" in os_environ else 2

        # TODO
        if 'dataset' in kwargs:
            dataset = kwargs['dataset']

        self.presets.update( { "base_model" : base_model.lower().replace("_","") } )
        self.presets.update( { "image_augmentation" : image_augmentation } )
        self.presets.update( { "reduce_lr_params" :  reduce_lr_params } )
        self.presets.update( { "validation_split" : validation_split } )
        self.presets.update( { "learning_rate" : learning_rate } )
        self.presets.update( { "batch_size" : batch_size } )
        self.presets.update( { "epochs" : epochs } )
        self.presets.update( { "freeze_layers" : freeze_layers } )
        self.presets.update( { "metrics" : metrics } )
        self.presets.update( { "checkpoint_monitor" : checkpoint_monitor } )
        self.presets.update( { "early_stopping_monitor" : early_stopping_monitor } )
        self.presets.update( { "early_stopping_patience" : early_stopping_patience } )
        self.presets.update( { "class_image_minimum" : class_image_minimum } )
        self.presets.update( { "class_image_maximum" : class_image_maximum } )
        self.presets.update( { "use_tensorboard" : use_tensorboard } )
        self.presets.update( { "use_imagenet_weights" : use_imagenet_weights } )
        self.presets.update( { "use_class_weights" : use_class_weights } )
        self.presets.update( { "image_list_class_column" : image_list_class_column } )
        self.presets.update( { "image_list_file_column" : image_list_file_column } )


    def get_preset(self, preset):
        if preset in self.presets:
            return self.presets[preset]
        else:
            raise ValueError("preset {} doesn't exist".format(preset))

