from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import pandas as pd
import numpy as np
from lib import logclass

# TODO: "WARNING:tensorflow:sample_weight modes were coerced from .. "
# https://stackoverflow.com/questions/59317919/warningtensorflowsample-weight-modes-were-coerced-from-to


def _csv_to_dataframe(filepath, usecols, encoding="utf-8-sig"):
    f = open(filepath, "r", encoding=encoding)
    line = f.readline()
    if line.count('\t') > 0:
        sep = '\t'
    else:
        sep = ','
    return pd.read_csv(filepath, encoding=encoding, sep=sep, dtype="str", usecols=usecols, header=None)


class ModelTrainer():
    project_root = None
    image_root = None
    downloaded_images_list_file = None
    image_list_class_col = None
    image_list_image_col = None
    class_list_file = None
    class_list_file_class_col = None
    model_save_path = None
    traindf = None
    class_list = None
    model_settings = None
    predictions = None

    COL_CLASS = "class"
    COL_IMAGE = "image"

    model = None
    train_generator = None
    validation_generator = None

    def __init__(self):
        self.logger = logclass.LogClass(self.__class__.__name__)
        self.logger.info("TensorFlow v{}".format(tf.__version__))

    def set_project_root(self, project_root, image_root=None):
        self.project_root = project_root

        if not os.path.isdir(self.project_root):
            raise FileNotFoundError("project root doesn't exist: {}".format(self.project_root))

        if image_root is not None:
            self.image_root = image_root
        else:
            self.image_root = os.path.join(self.project_root, 'images')

    def set_downloaded_images_list_file(self, downloaded_images_list_file=None, class_col=0, image_col=1):
        if downloaded_images_list_file is not None:
            self.downloaded_images_list_file = downloaded_images_list_file
        else:
            self.downloaded_images_list_file = os.path.join(self.project_root, 'lists', 'downloaded_images.csv')

        if not os.path.isfile(self.downloaded_images_list_file):
            raise FileNotFoundError("downloaded list file not found: {}".format(self.downloaded_images_list_file))

        self.image_list_class_col = class_col
        self.image_list_image_col = image_col

    def set_class_list_file(self, class_list_file=None, class_col=0):
        if class_list_file is not None:
            self.class_list_file = class_list_file
        else:
            self.class_list_file = os.path.join(self.project_root, 'lists', 'classes.csv')

        if not os.path.isfile(self.class_list_file):
            raise FileNotFoundError("class list file not found: {}".format(self.class_list_file))

        self.class_list_file_class_col = class_col

    # TODO: implement Test split
    def read_image_list_file(self):
        df = _csv_to_dataframe(filepath=self.downloaded_images_list_file,
                               usecols=[self.image_list_class_col, self.image_list_image_col])
        # if Test split
        #   df = df.sample(frac=1)
        #   msk = np.random.rand(len(df)) < 0.8
        #   self.traindf = df[msk]
        #   self.testdf = df[~msk]
        # # print(len(df), len(self.traindf), len(self.testdf))
        df[2] = self.image_root.rstrip("/") + "/" + df[2].astype(str)
        df.columns = [self.COL_CLASS, self.COL_IMAGE]
        self.traindf = df

    def read_class_list(self):
        self.class_list = _csv_to_dataframe(self.class_list_file, [self.class_list_file_class_col])

    def get_class_list(self):
        return self.class_list

    def get_model_save_path(self):
        self.model_save_path = os.path.join(self.project_root, 'models')
        return self.model_save_path

    def set_model_settings(self, model_settings):
        self.model_settings = model_settings
        for setting in self.model_settings:
            self.logger.info("setting - {}: {}".format(setting, str(self.model_settings[setting])))

    def configure_model(self):
        if "conv_base" in self.model_settings:
            conv_base = self.model_settings["conv_base"]
        else:
            conv_base = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False)

        x = conv_base.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)

        self.predictions = tf.keras.layers.Dense(len(self.class_list), activation='softmax')(x)
        self.model = tf.keras.models.Model(inputs=conv_base.input, outputs=self.predictions)
        self.model.compile(
            optimizer=self.model_settings["optimizer"],
            loss=self.model_settings["loss"],
            metrics=["acc"]
        )

    def configure_generators(self):
        a = self.model_settings["image_augmentation"] if "image_augmentation" in self.model_settings else []

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            validation_split=self.model_settings["validation_split"],
            rotation_range=a["rotation_range"] if "rotation_range" in a else 0,
            shear_range=a["shear_range"] if "shear_range" in a else 0.0,
            zoom_range=a["zoom_range"] if "zoom_range" in a else 0.0,
            width_shift_range=a["width_shift_range"] if "width_shift_range" in a else 0.0,
            height_shift_range=a["height_shift_range"] if "height_shift_range" in a else 0.0,
            horizontal_flip=a["horizontal_flip"] if "horizontal_flip" in a else False,
            vertical_flip=a["vertical_flip"] if "vertical_flip" in a else False
        )
            # preprocessing_function=self.preProcess,

        self.train_generator = datagen.flow_from_dataframe(
            dataframe=self.traindf,
            x_col=self.COL_IMAGE,
            y_col=self.COL_CLASS,
            class_mode="categorical",
            target_size=(299, 299),
            batch_size=self.model_settings["batch_size"],
            interpolation="nearest",
            subset="training")

        self.validation_generator = datagen.flow_from_dataframe(
            dataframe=self.traindf,
            x_col=self.COL_IMAGE,
            y_col=self.COL_CLASS,
            class_mode="categorical",
            target_size=(299, 299),
            batch_size=self.model_settings["batch_size"],
            interpolation="nearest",
            subset="validation")

    def train_model(self):
        step_size_train = self.train_generator.n // self.train_generator.batch_size
        step_size_validate = self.validation_generator.n // self.validation_generator.batch_size

        history = self.model.fit(
            x=self.train_generator,
            steps_per_epoch=step_size_train,
            epochs=self.model_settings["epochs"],
            validation_data=self.validation_generator,
            validation_steps=step_size_validate,
            callbacks=self.model_settings["callbacks"],
        )
        # If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets
        # will be obtained from x)


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.set_project_root(os.environ['PROJECT_ROOT'])

    trainer.set_downloaded_images_list_file(image_col=2)
    trainer.set_class_list_file()

    trainer.read_image_list_file()
    trainer.read_class_list()

    trainer.set_model_settings({
        "validation_split": 0.2,
        "conv_base": tf.keras.applications.InceptionV3(weights="imagenet", include_top=False),
        "batch_size": 64,
        "epochs": 200,
        "loss": "categorical_crossentropy",
        "optimizer": tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        "callbacks" : [ 
            tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, mode='auto', restore_best_weights=True),
            # tf.keras.callbacks.TensorBoard(),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss'),
            tf.keras.callbacks.ModelCheckpoint(trainer.get_model_save_path(), monitor='val_acc', save_best_only=True)
        ],
        "image_augmentation" : {
            "rotation_range": 90,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "horizontal_flip": True,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2, 
            "vertical_flip": false
        }
    })

    trainer.configure_model()
    trainer.configure_generators()
    trainer.train_model()
