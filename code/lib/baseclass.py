from __future__ import absolute_import, division, print_function, unicode_literals

import os, json
import tensorflow as tf
import pandas as pd
import numpy as np
from shutil import copyfile
from datetime import datetime
from lib import logclass


def _csv_to_dataframe(filepath, usecols, encoding="utf-8-sig"):
    f = open(filepath, "r", encoding=encoding)
    line = f.readline()
    if line.count('\t') > 0:
        sep = '\t'
    else:
        sep = ','
    return pd.read_csv(filepath, encoding=encoding, sep=sep, dtype="str", usecols=usecols, header=None)


class BaseClass():
    logger = None
    debug = False
    class_list_file_json = None
    class_list_file_csv = None
    image_list_file_csv = None
    downloaded_images_file = None
    timestamp = None
    model_name = None
    project_root = None
    image_path = None
    downloaded_images_list_file = None
    image_list_class_col = None
    image_list_image_col = None
    class_list_file = None
    class_list_file_class_col = None
    model_path = None
    architecture_path = None
    analysis_path = None
    traindf = None
    class_list = None
    model_settings = None
    COL_CLASS = "class"
    COL_IMAGE = "image"
    model = None

    def __init__(self):
        self.logger = logclass.LogClass(self.__class__.__name__)
        self.logger.info("TensorFlow v{}".format(tf.__version__))
        self.set_timestamp()
        # if self.debug:
        #     tf.get_logger().setLevel("DEBUG")
        #     tf.autograph.set_verbosity(10)
        # else:
        #     tf.get_logger().setLevel("INFO")
        #     tf.autograph.set_verbosity(1)

    def set_timestamp(self):
        d = datetime.now()
        self.timestamp = "{0}{1:02d}{2:02d}-{3:02d}{4:02d}{5:02d}".format(d.year,d.month,d.day,d.hour,d.minute,d.second)

    def set_debug(self,state):
        self.debug = state

    def set_model_name(self,model_name=None):
        if model_name is None:
            self.model_name = self.timestamp
        else:
            self.model_name = model_name
        self.logger.info("model name: {}".format(self.model_name))

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
        else:
            self.image_path = os.path.join(self.project_root, "images")

        if not os.path.exists(self.image_path):
            os.mkdir(self.image_path)
            self.logger.info("created image folder {}".format(self.image_path))

        self.downloaded_images_file = os.path.join(self.project_root, "lists", "downloaded_images.csv")
        self.class_list_file = os.path.join(self.project_root, "lists", "classes.csv")

    def copy_class_list_file(self):
        copyfile(self.class_list_file,self.class_list_file_model)

    def set_model_folder(self):
        self.model_folder = os.path.join(self.project_root, "models", self.model_name)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
            self.logger.info("created model folder {}".format(self.model_folder))

        self.class_list_file_model = os.path.join(self.model_folder, "classes.csv")

    def set_model_settings(self, model_settings):
        self.model_settings = model_settings
        for setting in self.model_settings:
            self.logger.info("setting - {}: {}".format(setting, str(self.model_settings[setting])))

    def set_downloaded_images_list_file(self, downloaded_images_list_file=None, class_col=0, image_col=1):
        if downloaded_images_list_file is not None:
            self.downloaded_images_list_file = downloaded_images_list_file

        if not os.path.isfile(self.downloaded_images_list_file):
            raise FileNotFoundError("downloaded list file not found: {}".format(self.downloaded_images_list_file))

        self.image_list_class_col = class_col
        self.image_list_image_col = image_col

    def set_class_list_file(self, class_list_file=None, class_col=0):
        if class_list_file is not None:
            self.class_list_file_model = class_list_file

        if not os.path.isfile(self.class_list_file_model):
            raise FileNotFoundError("class list file not found: {}".format(self.class_list_file_model))

        self.class_list_file_class_col = class_col

    # TODO: implement Test split
    def read_image_list_file(self):
        self.logger.info("reading images from: {}".format(self.downloaded_images_list_file))

        df = _csv_to_dataframe(filepath=self.downloaded_images_list_file,
                               usecols=[self.image_list_class_col, self.image_list_image_col])
        # if Test split
        #   df = df.sample(frac=1)
        #   msk = np.random.rand(len(df)) < 0.8
        #   self.traindf = df[msk]
        #   self.testdf = df[~msk]
        # # print(len(df), len(self.traindf), len(self.testdf))
        df[2] = self.image_path.rstrip("/") + "/" + df[2].astype(str)
        df.columns = [self.COL_CLASS, self.COL_IMAGE]
        self.traindf = df

        self.logger.info("read {} images in {} classes".format(self.traindf[self.COL_IMAGE].nunique(),self.traindf[self.COL_CLASS].nunique()))

    def read_class_list(self):
        self.class_list = _csv_to_dataframe(self.class_list_file_model, [self.class_list_file_class_col])

    def get_class_list(self):
        return self.class_list

    def get_model_path(self):
        self.model_path = os.path.join(self.model_folder, "model.hdf5")
        return self.model_path

    def get_architecture_path(self):
        self.architecture_path = os.path.join(self.model_folder, "architecture.json")
        return self.architecture_path

    def get_analysis_path(self):
        self.analysis_path = os.path.join(self.model_folder, "analysis_path.json")
        return self.analysis_path

    def get_classes_path(self):
        self.classes_save_path = os.path.join(self.model_folder, "classes.json")
        return self.classes_save_path

    def load_model(self):
        self.model = tf.keras.models.load_model(self.get_model_path(),compile=False)
        self.logger.info("loaded model {}".format(self.get_model_path()))

        with open(os.path.join(self.get_classes_path())) as f:
            self.classes = json.load(f)

        self.logger.info("loaded {} classes from {}".format(len(self.classes),self.get_classes_path()))

