from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from lib import logclass


def _csv_to_dataframe(filepath, usecols, encoding="utf-8-sig"):
    f = open(filepath, "r", encoding=encoding)
    line = f.readline()
    if line.count('\t') > 0:
        sep = '\t'
    else:
        sep = ','
    return pd.read_csv(filepath, encoding=encoding, sep=sep, dtype="str", usecols=usecols, header=None)


class ModelTrainer():
    debug = False
    project_root = None
    image_root = None
    downloaded_images_list_file = None
    image_list_class_col = None
    image_list_image_col = None
    class_list_file = None
    class_list_file_class_col = None
    model_save_path = None
    architecture_save_path = None
    traindf = None
    class_list = None
    model_settings = None
    predictions = None
    history = None
    training_phase = None
    current_freeze = None

    COL_CLASS = "class"
    COL_IMAGE = "image"

    model = None
    train_generator = None
    validation_generator = None

    def __init__(self):
        self.logger = logclass.LogClass(self.__class__.__name__)
        self.logger.info("TensorFlow v{}".format(tf.__version__))
        # if self.debug:
        #     tf.get_logger().setLevel("DEBUG")
        #     tf.autograph.set_verbosity(10)
        # else:
        #     tf.get_logger().setLevel("INFO")
        #     tf.autograph.set_verbosity(1)
        self.set_timestamp()

    def set_timestamp(self):
        d = datetime.now()
        self.timestamp = "{0}{1:02d}{2:02d}-{3:02d}{4:02d}{5:02d}".format(d.year,d.month,d.day,d.hour,d.minute,d.second)

    def set_debug(self,state):
        self.debug = state
        print(self.debug)

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
        self.logger.info("reading images from: {}".format(self.downloaded_images_list_file))

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

        self.logger.info("read {} images in {} classes".format(self.traindf[self.COL_IMAGE].nunique(),self.traindf[self.COL_CLASS].nunique()))

    def read_class_list(self):
        self.class_list = _csv_to_dataframe(self.class_list_file, [self.class_list_file_class_col])

    def get_class_list(self):
        return self.class_list

    def get_model_save_path(self):
        self.model_save_path = os.path.join(self.project_root, "models", self.timestamp + ".hdf5")
        return self.model_save_path

    def get_architecture_save_path(self):
        self.architecture_save_path = os.path.join(self.project_root, "models", self.timestamp + ".json")
        return self.architecture_save_path

    def get_history_plot_save_path(self):
        self.history_plot_save_path = os.path.join(self.project_root, "log", self.timestamp + ".png")
        return self.history_plot_save_path

    def get_tensorboard_log_path(self):
        self.tensorboard_log_path = os.path.join(self.project_root, "log", "logs_keras")
        return self.tensorboard_log_path

    def set_model_settings(self, model_settings):
        self.model_settings = model_settings
        for setting in self.model_settings:
            self.logger.info("setting - {}: {}".format(setting, str(self.model_settings[setting])))

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

        self.train_generator = datagen.flow_from_dataframe(
            dataframe=self.traindf,
            x_col=self.COL_IMAGE,
            y_col=self.COL_CLASS,
            class_mode="categorical",
            target_size=(299, 299),
            batch_size=self.model_settings["batch_size"],
            interpolation="nearest",
            subset="training",
            shuffle=True)

        self.validation_generator = datagen.flow_from_dataframe(
            dataframe=self.traindf,
            x_col=self.COL_IMAGE,
            y_col=self.COL_CLASS,
            class_mode="categorical",
            target_size=(299, 299),
            batch_size=self.model_settings["batch_size"],
            interpolation="nearest",
            subset="validation",
            shuffle=True)


    def assemble_model(self):
        if "base_model" in self.model_settings:
            self.base_model = self.model_settings["base_model"]
        else:
            self.base_model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False)

        x = self.base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.Dense(1024, activation='relu')(x)

        self.predictions = tf.keras.layers.Dense(len(self.class_list), activation='softmax')(x)
        self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=self.predictions)

        self.logger.info("model has {} layers (base model: {})".format(len(self.model.layers), len(self.base_model.layers)))


    def set_frozen_layers(self):
        self.model.trainable = True

        if not "freeze_layers" in self.model_settings:
            self.current_freeze="none specified"
            return

        print(self.training_phase)

        if isinstance(self.model_settings["freeze_layers"], list):
            print("a")
            if self.training_phase in self.model_settings["freeze_layers"]:
                self.current_freeze = self.model_settings["freeze_layers"][self.training_phase]
                print("b")

        else:
            self.current_freeze = self.model_settings["freeze_layers"]
            print("c")


        print(self.current_freeze)

        if self.current_freeze=="none":
            return

        if self.current_freeze=="base_model":
            self.base_model.trainable = False
        else:
            for layer in self.base_model.layers[:self.current_freeze]:
                layer.trainable = False


    def train_model(self):

        self.logger.info("start training {}".format(self.project_root))

        self.training_phase = 0

        if isinstance(self.model_settings["epochs"], int):
            epochs = [self.model_settings["epochs"]]
        else:
            epochs = self.model_settings["epochs"]

        for epoch in epochs: 

            self.logger.info("=== training phase {} ({}/{}) ===".format(self.training_phase,(self.training_phase+1),len(epochs)))

            self.set_frozen_layers()

            self.model.compile(
                optimizer=self.model_settings["optimizer"],
                loss=self.model_settings["loss"],
                metrics=self.model_settings["metrics"] if "metrics" in self.model_settings else [ "acc" ]
            )

            if self.debug:
                self.model.summary()
            else:
                self.logger.info("frozen layers: {}".format(self.current_freeze))

                params = self.get_trainable_params()

                self.logger.info("trainable variables: {:,}".format(len(self.model.trainable_variables)))
                self.logger.info("total parameters: {:,}".format(params["total"]))
                self.logger.info("trainable: {:,}".format(params["trainable"]))
                self.logger.info("non-trainable: {:,}".format(params["non_trainable"]))

                step_size_train = self.train_generator.n // self.train_generator.batch_size
                step_size_validate = self.validation_generator.n // self.validation_generator.batch_size

                self.history = self.model.fit(
                    x=self.train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=epoch,
                    validation_data=self.validation_generator,
                    validation_steps=step_size_validate,
                    callbacks=self.model_settings["callbacks"] if "callbacks" in self.model_settings else None
                )
                # If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets
                # will be obtained from x)

            self.training_phase += 1


        self.model.save(self.get_model_save_path())
        self.logger.info("saved model to {}".format(self.get_model_save_path()))

        f = open(self.get_architecture_save_path(), "w")
        f.write(self.model.to_json())
        f.close()
        self.logger.info("saved model architecture to {}".format(self.get_architecture_save_path()))





    def get_trainable_params(self):
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])

        return {
            "total" : trainable_count + non_trainable_count,
            "trainable" : trainable_count,
            "non_trainable" : non_trainable_count
        }


    def assemble_model_2(self):

        # Create the base model from the pre-trained model --> MobileNetV2
        self.base_model = tf.keras.applications.InceptionV3(include_top=False,
                                                       weights='imagenet')

        self.base_model.trainable = False
        self.base_model.summary()

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        # x = tf.keras.layers.Dense(1024, activation='relu')(x)
        # prediction_layer = tf.keras.layers.Dense(1)
        prediction_layer = tf.keras.layers.Dense(len(self.class_list), activation='softmax')

        self.model = tf.keras.Sequential([
          self.base_model,
          global_average_layer,
          prediction_layer
        ])

        # self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=self.predictions)


        base_learning_rate = 0.0001
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                      metrics=['acc'])


        print("\n\n====================== model 2 ===============================")

        self.model.summary()

        # self.model_settings["epochs"] = 10

        self.train_model()


        # self.base_model.trainable = True

        # # Let's take a look to see how many layers are in the base model
        # print("Number of layers in the base model: ", len(self.base_model.layers))

        # # Fine-tune from this layer onwards
        # fine_tune_at = 249

        # # Freeze all the layers before the `fine_tune_at` layer
        # for layer in self.base_model.layers[:fine_tune_at]:
        #     layer.trainable =  False

        # self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #               optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
        #               metrics=['acc'])

        # self.model.summary()

        # self.model_settings["epochs"] = 100

        # self.train_model()


    def evaluate(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(len(self.history.history["loss"]))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        # plt.show()
        plt.savefig(self.get_history_plot_save_path())


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    trainer.set_project_root(os.environ['PROJECT_ROOT'])
    trainer.set_downloaded_images_list_file(image_col=2)
    trainer.set_class_list_file()
    trainer.read_image_list_file()
    trainer.read_class_list()

        # "base_model": tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False),  
        # "base_model": tf.keras.applications.ResNet50(weights="imagenet", include_top=False),

    trainer.set_model_settings({
        "validation_split": 0.2,
        "base_model": tf.keras.applications.InceptionV3(weights="imagenet", include_top=False),  
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": tf.keras.optimizers.RMSprop(learning_rate=1e-4),
        "batch_size": 64,
        "epochs": [ 10, 100 ], # epochs single value or list controls whether training is phased
        "freeze_layers": [ "base_model", "none" ], # "base_model", # 249,
        "callbacks" : [ 
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, mode="auto", restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(trainer.get_tensorboard_log_path()),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=1e-8),
            tf.keras.callbacks.ModelCheckpoint(trainer.get_model_save_path(), monitor="val_acc", save_best_only=True, save_freq="epoch")
        ],
        "metrics" : [ "acc" ],
        "image_augmentation" : {
            "rotation_range": 90,
            "shear_range": 0.2,
            "zoom_range": 0.2,
            "horizontal_flip": True,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2, 
            "vertical_flip": False
        }
    })

    trainer.assemble_model()
    # trainer.configure_generators()
    # trainer.train_model()
    # trainer.evaluate()

    # trainer.configure_generators()
    # trainer.assemble_model_2()
    # trainer.train_model()
    # trainer.assemble_model()
    # trainer.train_model()
