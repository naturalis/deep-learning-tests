import os, json, argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from lib import baseclass, dataset, utils, customcallback


class ModelTrainer(baseclass.BaseClass):
    timestamp = None
    predictions = None
    history = []
    training_phase = None
    current_freeze = None
    current_optimizer = None

    train_generator = None
    validation_generator = None
    current_epoch = 0

    customcallback = None

    def __init__(self):
        super().__init__()

    def get_tensorboard_log_path(self):
        self.tensorboard_log_path = os.path.join(self.project_root, "log", "logs_keras")
        return self.tensorboard_log_path

    def get_trainable_params(self):
        trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])

        return {
            "total" : trainable_count + non_trainable_count,
            "trainable" : trainable_count,
            "non_trainable" : non_trainable_count
        }

    def configure_optimizers(self):
        optimizers = []
        for learning_rate in self.get_preset("learning_rate"):
            optimizers.append(tf.keras.optimizers.RMSprop(learning_rate=learning_rate))

        return optimizers

    def configure_callbacks(self):
        self.customcallback = customcallback.CustomCallback()

        callbacks = []
        for key,epoch in enumerate(self.get_preset("epochs")):
            phase = []

            phase.append(self.customcallback)

            phase.append(tf.keras.callbacks.ModelCheckpoint(self.get_model_path(), 
                monitor=self.get_preset("checkpoint_monitor"), save_best_only=True, save_freq="epoch", verbose=1))

            if self.get_preset("use_tensorboard"):
                phase.append(tf.keras.callbacks.TensorBoard(self.get_tensorboard_log_path()))
                self.logger.info("tensor board log path: {}".format(self.get_tensorboard_log_path()))

            if key < len(self.get_preset("early_stopping_monitor")):
                item = self.get_preset("early_stopping_monitor")[key]
                if not item == "none":
                    phase.append(tf.keras.callbacks.EarlyStopping(
                        monitor=item, 
                        patience=5, 
                        mode="auto", 
                        restore_best_weights=True, 
                        verbose=1))

            if key < len(self.get_preset("reduce_lr_params")):
                item = self.get_preset("reduce_lr_params")[key]
                phase.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=item["monitor"],
                    factor=item["factor"],
                    patience=item["patience"],
                    min_lr=item["min_lr"],
                    verbose=item["verbose"]))

            callbacks.append(phase)

        return callbacks

        # for item in trainer.get_preset("reduce_lr_params"):
        #     phase = []
        #     phase.append(tf.keras.callbacks.EarlyStopping(
        #         monitor=trainer.get_preset("early_stopping_monitor"), patience=5, mode="auto", restore_best_weights=True, verbose=1))
        #     phase.append(tf.keras.callbacks.ModelCheckpoint(trainer.get_model_path(), 
        #         monitor=trainer.get_preset("checkpoint_monitor"), save_best_only=True, save_freq="epoch", verbose=1))
        #     phase.append(tf.keras.callbacks.ReduceLROnPlateau(
        #         monitor=item["monitor"],factor=item["factor"],patience=item["patience"],min_lr=item["min_lr"],verbose=item["verbose"]))
        #     callbacks.append(phase)

        # return callbacks

    def configure_generators(self):
        if "image_augmentation" in self.model_settings and not self.model_settings["image_augmentation"] is None:
            a = self.model_settings["image_augmentation"]
        else:
            a = []

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
        if "base_model" in self.model_settings and self.model_settings["base_model"]=="custom":
            self._assemble_custom_model()
        else:
            self._assemble_transfer_model()


    def _assemble_custom_model(self):

        self.logger.info("using custom model")

        input_shape = (299, 299, 3)

        x = tf.keras.layers.Conv2D(16, (3,3), padding='same', input_shape=input_shape, activation='relu')
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        self.predictions = tf.keras.layers.Dense(len(self.class_list), activation='softmax')(x)

        self.model = tf.keras.models.Model(inputs=input_shape, outputs=self.predictions)

        self.logger.info("model has {} classes".format(len(self.class_list)))
        self.logger.info("model has {} layers".format(len(self.model.layers)))



    def _assemble_transfer_model(self):
        
        self.base_model = None

        if "base_model" in self.model_settings:

            if self.model_settings["base_model"] == "mobilenetv2":
                self.base_model = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False)

            if self.model_settings["base_model"] == "resnet50":
                self.base_model = tf.keras.applications.ResNet50(weights="imagenet", include_top=False)

            if self.model_settings["base_model"] == "vgg16":
                self.base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)

            if self.model_settings["base_model"] == "inceptionresnetv2":
                self.base_model = tf.keras.applications.InceptionResNetV2(weights="imagenet", include_top=False)

            if self.model_settings["base_model"] == "inceptionv3":
                self.base_model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False)

            if self.model_settings["base_model"] == "xception":
                # self.base_model = tf.keras.applications.Xception(weights="imagenet", include_top=False)
                self.base_model = tf.keras.applications.Xception(weights=None, include_top=False)

            if self.base_model == None:
                self.logger.error("unknown base model: {}".format(self.model_settings["base_model"]))

        if self.base_model == None:
            self.base_model = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False)

        self.logger.info("using base model {}".format(self.base_model.name))

        x = self.base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # extra layers
        # x = tf.keras.layers.Dense(512, activation='relu')(x)
        # x = tf.keras.layers.Dense(1024, activation='relu')(x)
        # x = tf.keras.layers.Dropout(.2)(x)

        self.predictions = tf.keras.layers.Dense(len(self.class_list), activation='softmax')(x)
        self.model = tf.keras.models.Model(inputs=self.base_model.input, outputs=self.predictions)

        self.logger.info("model has {} classes".format(len(self.class_list)))
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

    def set_current_optimizer(self):
        if not "optimizer" in self.model_settings:
            self.current_optimizer = None
            return None

        if isinstance(self.model_settings["optimizer"], list):
            if self.training_phase < len(self.model_settings["optimizer"]):
                self.current_optimizer = self.model_settings["optimizer"][self.training_phase]
        else:
            self.current_optimizer = self.model_settings["optimizer"]

    def set_current_callbacks(self):
        if not "callbacks" in self.model_settings:
            self.current_callbacks = None

        if isinstance(self.model_settings["callbacks"], list):
            if self.training_phase < len(self.model_settings["callbacks"]):
                self.current_callbacks = self.model_settings["callbacks"][self.training_phase]
        else:
            self.current_callbacks = self.model_settings["callbacks"]

    def train_model(self):

        self.logger.info("start training \"{}\" ({})".format(self.project_name,self.project_root))

        self.training_phase = 0

        if isinstance(self.model_settings["epochs"], int):
            self.epochs = [self.model_settings["epochs"]]
        else:
            self.epochs = self.model_settings["epochs"]

        for epoch in self.epochs: 

            self.logger.info("=== training phase {}/{} ===".format((self.training_phase+1),len(self.epochs)))

            self.set_frozen_layers()
            self.set_current_optimizer()

            self.model.compile(
                optimizer=self.current_optimizer,
                loss=self.model_settings["loss"],
                metrics=self.model_settings["metrics"] if "metrics" in self.model_settings else [ "acc","loss","val_acc","val_loss" ]
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

            self.set_current_callbacks()

            history = self.model.fit(
                x=self.train_generator,
                steps_per_epoch=step_size_train,
                epochs=epoch,
                validation_data=self.validation_generator,
                validation_steps=step_size_validate,
                callbacks=self.current_callbacks
            )

            # If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets
            # will be obtained from x)

            self.history.append(history)
            self.training_phase += 1

    def save_model(self):
        self.model.save(self.get_model_path())
        self.logger.info("saved model to {}".format(self.get_model_path()))

    def save_model_architecture(self):
        f = open(self.get_architecture_path(), "w")
        f.write(self.model.to_json())
        f.close()
        self.logger.info("saved model architecture to {}".format(self.get_architecture_path()))

    def save_history(self):
        for phase,hist in enumerate(self.history):
            self.save_history_plot(phase)
            # path = os.path.join(self.model_folder, "history_phase_{}.csv".format(phase))
            # self.logger.info("saved history {}".format(path))

    def save_history_plot(self,phase=None):
        if phase is None:
            phase = self.training_phase

        acc = self.history[phase].history['acc']
        val_acc = self.history[phase].history['val_acc']

        loss = self.history[phase].history['loss']
        val_loss = self.history[phase].history['val_loss']

        epochs_range = range(len(self.history[phase].history["loss"]))

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

        path = os.path.join(self.model_folder, "history_plot_phase_{}.png".format(phase))
        plt.savefig(path)
        self.logger.info("saved plot {}".format(path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--dataset_note",type=str)
    parser.add_argument("--load_model",type=str)
    args = parser.parse_args() 

    trainer = ModelTrainer()
    timer = utils.Timer()
    dataset = dataset.DataSet()

    dataset.set_environ(os.environ)
    trainer.set_debug(os.environ["DEBUG"]=="1" if "DEBUG" in os.environ else False)    
    trainer.set_project(os.environ)
    trainer.set_presets(os_environ=os.environ)
    trainer.set_class_image_minimum(trainer.get_preset("class_image_minimum"))
    trainer.set_class_image_maximum(trainer.get_preset("class_image_maximum"))

    if args.load_model: 
        trainer.set_model_name(args.load_model)
    else:
        trainer.set_model_name(trainer.make_model_name())

    trainer.set_model_folder()

    trainer.set_model_settings({
        "validation_split": trainer.get_preset("validation_split"),
        "base_model": trainer.get_preset("base_model"),
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": trainer.configure_optimizers(),
        "batch_size": trainer.get_preset("batch_size"),
        "epochs": trainer.get_preset("epochs"), 
        "freeze_layers": trainer.get_preset("freeze_layers"), 
        "callbacks" : trainer.configure_callbacks(),
        "metrics" : trainer.get_preset("metrics"),
        "image_augmentation" : trainer.get_preset("image_augmentation")
    })

    if args.dataset_note: 
        dataset.set_note(args.dataset_note)

    if args.load_model: 

        del trainer.model_settings["freeze_layers"]

        trainer.load_model()
        trainer.read_class_list()
        trainer.read_image_list_file(image_col=2)
        trainer.image_list_apply_class_list()
        dataset.ask_retrain_note()
        dataset.set_model_trainer(trainer)
        dataset.open_dataset()

    else:

        trainer.copy_image_list_file()
        trainer.copy_class_list_file()

        trainer.read_image_list_file(image_col=2)
        trainer.read_class_list()

        trainer.image_list_apply_class_list()
        trainer.assemble_model()
        trainer.save_model_architecture()
        dataset.ask_note()
        dataset.set_model_trainer(trainer)
        dataset.make_dataset()

    dataset.update_model_state("training")
    dataset.save_dataset()

    trainer.configure_generators()
    trainer.train_model()

    dataset.set_epochs_trained(trainer.customcallback.get_current_epoch())
    dataset.set_training_time(timer.get_time_passed())
    dataset.update_model_state("configured")

    dataset.save_dataset()

    trainer.save_model()
    trainer.save_history()


        # WARNING:tensorflow:Method (on_train_batch_end) is slow compared to the batch update (0.325404). Check your callbacks.
        # maybe something with TensorBoard callback, as the other ones get called at epoch end, not batch end

        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler

