import os, json
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from lib import baseclass

class ModelTrainer(baseclass.BaseClass):
    timestamp = None
    predictions = None
    history = None
    training_phase = None
    current_freeze = None
    current_optimizer = None

    train_generator = None
    validation_generator = None

    def __init__(self):
        super().__init__()

    def get_history_plot_save_path(self):
        self.history_plot_save_path = os.path.join(self.project_root, "log", self.timestamp + ".png")
        return self.history_plot_save_path

    def get_tensorboard_log_path(self):
        self.tensorboard_log_path = os.path.join(self.project_root, "log", "logs_keras")
        return self.tensorboard_log_path

    def configure_generators(self):
        a = self.model_settings["image_augmentation"] if "image_augmentation" in self.model_settings else []

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
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
            shuffle=True
        )

        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            validation_split=self.model_settings["validation_split"],
        )

        self.validation_generator = datagen.flow_from_dataframe(
            dataframe=self.traindf,
            x_col=self.COL_IMAGE,
            y_col=self.COL_CLASS,
            class_mode="categorical",
            target_size=(299, 299),
            batch_size=self.model_settings["batch_size"],
            interpolation="nearest",
            subset="validation",
            shuffle=True
        )

        f = open(self.get_classes_path(), "w")
        f.write(json.dumps(self.train_generator.class_indices))
        f.close()

        self.logger.info("saved model classes to {}".format(self.get_classes_path()))

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
            self.current_freeze="none"
            return

        if isinstance(self.model_settings["freeze_layers"], list):
            if self.training_phase < len(self.model_settings["freeze_layers"]):
                self.current_freeze = self.model_settings["freeze_layers"][self.training_phase]
        else:
            self.current_freeze = self.model_settings["freeze_layers"]

        if self.current_freeze=="none":
            return

        if self.current_freeze=="base_model":
            self.base_model.trainable = False
        else:
            for layer in self.base_model.layers[:self.current_freeze]:
                layer.trainable = False


    def set_callbacks(self):
        if not "callbacks" in self.model_settings:
            self.current_callbacks = None
            return None

        if isinstance(self.model_settings["callbacks"], list):
            if self.training_phase < len(self.model_settings["callbacks"]):
                self.current_callbacks = self.model_settings["callbacks"][self.training_phase]
        else:
            self.current_callbacks = self.model_settings["callbacks"]


    def set_optimizer(self):
        if not "optimizer" in self.model_settings:
            self.current_optimizer = None
            return None

        if isinstance(self.model_settings["optimizer"], list):
            if self.training_phase < len(self.model_settings["optimizer"]):
                self.current_optimizer = self.model_settings["optimizer"][self.training_phase]
        else:
            self.current_optimizer = self.model_settings["optimizer"]


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
            self.set_optimizer()

            self.model.compile(
                optimizer=self.current_optimizer,
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

            self.set_callbacks()

            self.history = self.model.fit(
                x=self.train_generator,
                steps_per_epoch=step_size_train,
                epochs=epoch,
                validation_data=self.validation_generator,
                validation_steps=step_size_validate,
                callbacks=self.current_callbacks
            )

            # If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets
            # will be obtained from x)

            self.training_phase += 1


    def save_model(self):
        self.model.save(self.get_model_path())
        self.logger.info("saved model to {}".format(self.get_model_path()))

        f = open(self.get_architecture_path(), "w")
        f.write(self.model.to_json())
        f.close()
        self.logger.info("saved model architecture to {}".format(self.get_architecture_path()))


    def get_trainable_params(self):
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])

        return {
            "total" : trainable_count + non_trainable_count,
            "trainable" : trainable_count,
            "non_trainable" : non_trainable_count
        }


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
    trainer.set_model_name()
    trainer.set_project_folders(project_root=os.environ['PROJECT_ROOT'])
    trainer.set_downloaded_images_list_file(image_col=2)
    trainer.set_class_list_file()
    trainer.read_image_list_file()
    trainer.read_class_list()

        # "base_model": tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False),  
        # "base_model": tf.keras.applications.ResNet50(weights="imagenet", include_top=False),

        # WARNING:tensorflow:Method (on_train_batch_end) is slow compared to the batch update (0.325404). Check your callbacks.
        # maybe something with TensorBoard callback, as the other ones get called at epoch end, not batch end


        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler

    trainer.set_model_settings({
        "validation_split": 0.2,
        "base_model": tf.keras.applications.InceptionV3(weights="imagenet", include_top=False),  
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": [
            tf.keras.optimizers.RMSprop(learning_rate=1e-5),
        ],
        "batch_size": 64,
        "epochs": [ 200 ], # epochs single value or list controls whether training is phased
        "freeze_layers": [ "none" ], # "base_model", # 249, # none
        "callbacks" : [
            [ 
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="auto", restore_best_weights=True, verbose=1),
                # tf.keras.callbacks.TensorBoard(trainer.get_tensorboard_log_path()),
                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=1e-8, verbose=1),
                tf.keras.callbacks.ModelCheckpoint(trainer.get_model_path(), monitor="val_acc", save_best_only=True, save_freq="epoch", verbose=1)
            ]
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
    trainer.configure_generators()
    trainer.train_model()
    trainer.save_model()
    trainer.evaluate()

