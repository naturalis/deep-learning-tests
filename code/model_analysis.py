import os, sys
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
from lib import baseclass

class ModelAnalysis(baseclass.BaseClass):
    test_generator = None

    def __init__(self):
        super().__init__()

    def configure_generator(self):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
        )

        self.test_generator = datagen.flow_from_dataframe(
            dataframe=self.traindf,
            x_col=self.COL_IMAGE,
            y_col=self.COL_CLASS,
            class_mode="categorical",
            target_size=(299, 299),
            batch_size=self.model_settings["batch_size"],
            interpolation="nearest",
            shuffle=False
        )

    def do_stuff(self):
        batch_size = self.model_settings["batch_size"]

        Y_pred = self.model.predict(self.test_generator)
        y_pred = np.argmax(Y_pred, axis=1)

        # self.logger.info("saved model to {}".format(self.get_model_path()))

        print('Confusion Matrix')
        cm = tf.math.confusion_matrix(self.test_generator.classes, y_pred)
        for row in cm:
            print("[ ",end='')
            for cell in row:
                print("{:>6d}".format(cell),end='')
            print(" ]")

        print('Classification Report')
        cp = classification_report(self.test_generator.classes, y_pred)
        print(cp)

        f = open(self.get_analysis_path(), "w")
        f.write(json.dumps({"confusion matrix" : cm, "classification report" : cp})
        f.close()


if __name__ == "__main__":

    project_root = os.environ['PROJECT_ROOT']

    if len(sys.argv)>1:
        model_name = sys.argv[1]
    else:
        model_name = os.environ['MODEL_NAME']

    analysis = ModelAnalysis()

    analysis.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)
    analysis.set_model_name(model_name)
    analysis.set_project_folders(project_root=project_root)

    analysis.load_model()

    analysis.set_downloaded_images_list_file(image_col=2)
    analysis.set_class_list_file()
    analysis.read_image_list_file()
    analysis.read_class_list()

    analysis.set_model_settings({
        "batch_size": 64,
    })

    analysis.configure_generator()
    analysis.do_stuff()









    # def set_downloaded_images_list_file(self, downloaded_images_list_file=None, class_col=0, image_col=1):
    #     if downloaded_images_list_file is not None:
    #         self.downloaded_images_list_file = downloaded_images_list_file
    #     else:
    #         self.downloaded_images_list_file = os.path.join(self.project_root, 'lists', 'downloaded_images.csv')

    #     if not os.path.isfile(self.downloaded_images_list_file):
    #         raise FileNotFoundError("downloaded list file not found: {}".format(self.downloaded_images_list_file))

    #     self.image_list_class_col = class_col
    #     self.image_list_image_col = image_col

    # def set_class_list_file(self, class_list_file=None, class_col=0):
    #     if class_list_file is not None:
    #         self.class_list_file = class_list_file
    #     else:
    #         self.class_list_file = os.path.join(self.project_root, 'lists', 'classes.csv')

    #     if not os.path.isfile(self.class_list_file):
    #         raise FileNotFoundError("class list file not found: {}".format(self.class_list_file))

    #     self.class_list_file_class_col = class_col

    # # TODO: implement Test split
    # def read_image_list_file(self):
    #     self.logger.info("reading images from: {}".format(self.downloaded_images_list_file))

    #     df = _csv_to_dataframe(filepath=self.downloaded_images_list_file,
    #                            usecols=[self.image_list_class_col, self.image_list_image_col])
    #     # if Test split
    #     #   df = df.sample(frac=1)
    #     #   msk = np.random.rand(len(df)) < 0.8
    #     #   self.traindf = df[msk]
    #     #   self.testdf = df[~msk]
    #     # # print(len(df), len(self.traindf), len(self.testdf))
    #     df[2] = self.image_root.rstrip("/") + "/" + df[2].astype(str)
    #     df.columns = [self.COL_CLASS, self.COL_IMAGE]
    #     self.traindf = df

    #     self.logger.info("read {} images in {} classes".format(self.traindf[self.COL_IMAGE].nunique(),self.traindf[self.COL_CLASS].nunique()))

    # def read_class_list(self):
    #     self.class_list = _csv_to_dataframe(self.class_list_file, [self.class_list_file_class_col])

    # def get_class_list(self):
    #     return self.class_list

    # def get_model_save_path(self):
    #     self.model_save_path = os.path.join(self.model_folder, "model.hdf5")
    #     return self.model_save_path

    # def get_architecture_save_path(self):
    #     self.architecture_save_path = os.path.join(self.model_folder, "architecture.json")
    #     return self.architecture_save_path

    # def get_classes_save_path(self):
    #     self.classes_save_path = os.path.join(self.model_folder, "classes.json")
    #     return self.classes_save_path

    # def get_history_plot_save_path(self):
    #     self.history_plot_save_path = os.path.join(self.project_root, "log", self.timestamp + ".png")
    #     return self.history_plot_save_path

    # def get_tensorboard_log_path(self):
    #     self.tensorboard_log_path = os.path.join(self.project_root, "log", "logs_keras")
    #     return self.tensorboard_log_path

    # def set_model_settings(self, model_settings):
    #     self.model_settings = model_settings
    #     for setting in self.model_settings:
    #         self.logger.info("setting - {}: {}".format(setting, str(self.model_settings[setting])))

    # def configure_generators(self):
    #     a = self.model_settings["image_augmentation"] if "image_augmentation" in self.model_settings else []

    #     datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    #         validation_split=self.model_settings["validation_split"],
    #         rotation_range=a["rotation_range"] if "rotation_range" in a else 0,
    #         shear_range=a["shear_range"] if "shear_range" in a else 0.0,
    #         zoom_range=a["zoom_range"] if "zoom_range" in a else 0.0,
    #         width_shift_range=a["width_shift_range"] if "width_shift_range" in a else 0.0,
    #         height_shift_range=a["height_shift_range"] if "height_shift_range" in a else 0.0,
    #         horizontal_flip=a["horizontal_flip"] if "horizontal_flip" in a else False,
    #         vertical_flip=a["vertical_flip"] if "vertical_flip" in a else False
    #     )

    #     self.train_generator = datagen.flow_from_dataframe(
    #         dataframe=self.traindf,
    #         x_col=self.COL_IMAGE,
    #         y_col=self.COL_CLASS,
    #         class_mode="categorical",
    #         target_size=(299, 299),
    #         batch_size=self.model_settings["batch_size"],
    #         interpolation="nearest",
    #         subset="training",
    #         shuffle=True)

    #     self.validation_generator = datagen.flow_from_dataframe(
    #         dataframe=self.traindf,
    #         x_col=self.COL_IMAGE,
    #         y_col=self.COL_CLASS,
    #         class_mode="categorical",
    #         target_size=(299, 299),
    #         batch_size=self.model_settings["batch_size"],
    #         interpolation="nearest",
    #         subset="validation",
    #         shuffle=True)

    #     f = open(self.get_classes_save_path(), "w")
    #     f.write(json.dumps(self.train_generator.class_indices))
    #     f.close()

    #     self.logger.info("saved model classes to {}".format(self.get_classes_save_path()))


    # def assemble_model(self):
    #     if "base_model" in self.model_settings:
    #         self.base_model = self.model_settings["base_model"]
    #     else:
    #         self.base_model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False)

    #     x = self.base_model.output
    #     x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #     # x = tf.keras.layers.Dense(1024, activation='relu')(x)

    #     self.predictions = tf.keras.layers.Dense(len(self.class_list), activation='softmax')(x)
    #     self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=self.predictions)

    #     self.logger.info("model has {} layers (base model: {})".format(len(self.model.layers), len(self.base_model.layers)))


    # def set_frozen_layers(self):
    #     self.model.trainable = True

    #     if not "freeze_layers" in self.model_settings:
    #         self.current_freeze="none"
    #         return

    #     if isinstance(self.model_settings["freeze_layers"], list):
    #         if self.training_phase < len(self.model_settings["freeze_layers"]):
    #             self.current_freeze = self.model_settings["freeze_layers"][self.training_phase]
    #     else:
    #         self.current_freeze = self.model_settings["freeze_layers"]

    #     if self.current_freeze=="none":
    #         return

    #     if self.current_freeze=="base_model":
    #         self.base_model.trainable = False
    #     else:
    #         for layer in self.base_model.layers[:self.current_freeze]:
    #             layer.trainable = False


    # def set_callbacks(self):
    #     if not "callbacks" in self.model_settings:
    #         self.current_callbacks = None
    #         return None

    #     if isinstance(self.model_settings["callbacks"], list):
    #         if self.training_phase < len(self.model_settings["callbacks"]):
    #             self.current_callbacks = self.model_settings["callbacks"][self.training_phase]
    #     else:
    #         self.current_callbacks = self.model_settings["callbacks"]


    # def train_model(self):

    #     self.logger.info("start training {}".format(self.project_root))

    #     self.training_phase = 0

    #     if isinstance(self.model_settings["epochs"], int):
    #         epochs = [self.model_settings["epochs"]]
    #     else:
    #         epochs = self.model_settings["epochs"]

    #     for epoch in epochs: 

    #         self.logger.info("=== training phase {} ({}/{}) ===".format(self.training_phase,(self.training_phase+1),len(epochs)))

    #         self.set_frozen_layers()

    #         self.model.compile(
    #             optimizer=self.model_settings["optimizer"],
    #             loss=self.model_settings["loss"],
    #             metrics=self.model_settings["metrics"] if "metrics" in self.model_settings else [ "acc" ]
    #         )

    #         if self.debug:
    #             self.model.summary()
    #         else:
    #             self.logger.info("frozen layers: {}".format(self.current_freeze))

    #             params = self.get_trainable_params()

    #             self.logger.info("trainable variables: {:,}".format(len(self.model.trainable_variables)))
    #             self.logger.info("total parameters: {:,}".format(params["total"]))
    #             self.logger.info("trainable: {:,}".format(params["trainable"]))
    #             self.logger.info("non-trainable: {:,}".format(params["non_trainable"]))

    #         step_size_train = self.train_generator.n // self.train_generator.batch_size
    #         step_size_validate = self.validation_generator.n // self.validation_generator.batch_size

    #         self.set_callbacks()

    #         self.history = self.model.fit(
    #             x=self.train_generator,
    #             steps_per_epoch=step_size_train,
    #             epochs=epoch,
    #             validation_data=self.validation_generator,
    #             validation_steps=step_size_validate,
    #             callbacks=self.current_callbacks
    #         )

    #         # If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets
    #         # will be obtained from x)

    #         self.training_phase += 1


    # def save_model(self):
    #     self.model.save(self.get_model_save_path())
    #     self.logger.info("saved model to {}".format(self.get_model_save_path()))

    #     f = open(self.get_architecture_save_path(), "w")
    #     f.write(self.model.to_json())
    #     f.close()
    #     self.logger.info("saved model architecture to {}".format(self.get_architecture_save_path()))


    # def get_trainable_params(self):
    #     trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
    #     non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])

    #     return {
    #         "total" : trainable_count + non_trainable_count,
    #         "trainable" : trainable_count,
    #         "non_trainable" : non_trainable_count
    #     }


    # def evaluate(self):
    #     acc = self.history.history['acc']
    #     val_acc = self.history.history['val_acc']

    #     loss = self.history.history['loss']
    #     val_loss = self.history.history['val_loss']

    #     epochs_range = range(len(self.history.history["loss"]))

    #     plt.figure(figsize=(8, 8))
    #     plt.subplot(1, 2, 1)
    #     plt.plot(epochs_range, acc, label='Training Accuracy')
    #     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    #     plt.legend(loc='lower right')
    #     plt.title('Training and Validation Accuracy')

    #     plt.subplot(1, 2, 2)
    #     plt.plot(epochs_range, loss, label='Training Loss')
    #     plt.plot(epochs_range, val_loss, label='Validation Loss')
    #     plt.legend(loc='upper right')
    #     plt.title('Training and Validation Loss')
    #     # plt.show()
    #     plt.savefig(self.get_history_plot_save_path())
