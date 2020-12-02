import os, json, argparse
import tensorflow as tf
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import class_weight, resample
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
    class_weight = None

    customcallback = None
    skip_training = False

    am_training = None
    dataset = None
    timer = None

    def __init__(self):
        super().__init__()

    def __del__(self): 
        print("-------------> interrupted_training")
        # script interrupted during training
        if self.am_training==True:
            self.logger.info("training interrupted")
            if self.dataset:
                self.dataset.set_epochs_trained(self.customcallback.get_current_epoch())
                if self.timer:
                    self.dataset.set_training_time(self.timer.get_time_passed())
                self.dataset.update_model_state("interrupted_training")
                self.dataset.save_dataset()

            self.save_model()
            self.save_history()

    def set_skip_training(self,skip_training):
        self.skip_training = skip_training

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

    def set_dataset(self,dataset):
        self.dataset = dataset

    def set_timer(self,timer):
        self.timer = timer

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
                monitor = self.get_preset("early_stopping_monitor")[key]
                patience = self.get_preset("early_stopping_patience")[key]
                if not monitor == "none":
                    phase.append(tf.keras.callbacks.EarlyStopping(
                        monitor=monitor, 
                        patience=patience,
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


    def get_imbalance_ratio(self):
        counts = self.traindf[self.COL_CLASS].value_counts()
        return min(counts) / max(counts)

    def upsample(self, factor=1.1):

        target_ratio = self.get_preset("upsampling_ratio")

        self.logger.info("upsampling, target ratio: {}".format(target_ratio))

        while(self.get_imbalance_ratio() < target_ratio):   
            counts = self.traindf[self.COL_CLASS].value_counts()
            minority_label = counts.idxmin()
            dataset_minority = self.traindf[self.traindf[self.COL_CLASS]==minority_label]
            dataset_other = self.traindf[self.traindf[self.COL_CLASS]!=minority_label]
            dataset_minority_upsampled = resample(dataset_minority,
                                                    replace=True,
                                                    n_samples=int(np.ceil(len(dataset_minority) * factor)),
                                                    random_state=111)

            self.traindf = pd.concat([dataset_minority_upsampled, dataset_other])
            self.logger.info("upsampling, dataset num rows: {}".format(len(self.traindf)))

    def configure_generators(self):

        if "image_augmentation" in self.model_settings and not self.model_settings["image_augmentation"] is None:
            a = self.model_settings["image_augmentation"]
        else:
            a = []
        
        if "upsampling_ratio" in self.model_settings and not self.model_settings["upsampling_ratio"] == -1:
            self.upsample()

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

        if self.get_preset("use_imagenet_weights"):
            weights = "imagenet"
        else:
            weights = None

        if "base_model" in self.model_settings:

            if self.model_settings["base_model"] == "mobilenetv2":
                self.base_model = tf.keras.applications.MobileNetV2(weights=weights, include_top=False)

            if self.model_settings["base_model"] == "resnet50":
                self.base_model = tf.keras.applications.ResNet50(weights=weights, include_top=False)

            if self.model_settings["base_model"] == "vgg16":
                self.base_model = tf.keras.applications.VGG16(weights=weights, include_top=False)

            if self.model_settings["base_model"] == "inceptionresnetv2":
                self.base_model = tf.keras.applications.InceptionResNetV2(weights=weights, include_top=False)

            if self.model_settings["base_model"] == "inceptionv3":
                self.base_model = tf.keras.applications.InceptionV3(weights=weights, include_top=False)

            if self.model_settings["base_model"] == "xception":
                self.base_model = tf.keras.applications.Xception(weights=weights, include_top=False)

            if self.base_model == None:
                self.logger.error("unknown base model: {}".format(self.model_settings["base_model"]))

        if self.base_model == None:
            self.base_model = tf.keras.applications.InceptionV3(weights=weights, include_top=False)

        self.logger.info("base model: {}".format(self.base_model.name))
        self.logger.info("use imagenet weights: {}".format("y" if self.get_preset("use_imagenet_weights") else "n"))

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

        if "use_class_weights" in self.model_settings and not self.model_settings["use_class_weights"] is False:
            total = np.sum([int(b[1]) for b in self.class_list ])
            self.class_weight = {}
            for k,item in enumerate(self.class_list):
                self.class_weight[k]=(1 / int(item[1]))*(total)/2.0

            self.logger.info("using class weights")
            self.logger.info(self.class_weight)
            self.logger.info(self.class_list)
            print(self.train_generator.class_indices)

        else:
            self.class_weight = None

            self.logger.info("using no class weights")


        self.logger.info("start training \"{}\" ({})".format(self.project_name,self.project_root))

        self.training_phase = 0

        if isinstance(self.model_settings["epochs"], int):
            self.epochs = [self.model_settings["epochs"]]
        else:
            self.epochs = self.model_settings["epochs"]

        self.am_training = True

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

            if not self.skip_training:

                history = self.model.fit(
                    x=self.train_generator,
                    steps_per_epoch=step_size_train,
                    epochs=epoch,
                    validation_data=self.validation_generator,
                    validation_steps=step_size_validate,
                    callbacks=self.current_callbacks,
                    class_weight=self.class_weight
                )

                # If x is a dataset, generator, or keras.utils.Sequence instance, y should not be specified (since targets
                # will be obtained from x)

                # fit:
                # initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).

                self.history.append(history)

            else:

                self.logger.info("skipping training (--no_training)")

            self.training_phase += 1

        self.am_training = False


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

        plt.figure(figsize=(24, 8)) # WxH
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
    parser.add_argument("--alt_class_list",type=str,help="specify alternative class list")
    parser.add_argument("--skip_training",action="store_true")
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

    trainer.set_skip_training(args.skip_training)

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
        "image_augmentation" : trainer.get_preset("image_augmentation"),
        "use_class_weights" : trainer.get_preset("use_class_weights"),
        "upsampling_ratio" : trainer.get_preset("upsampling_ratio")
    })

    if args.dataset_note: 
        dataset.set_note(args.dataset_note)

    if args.alt_class_list: 
        trainer.set_alternative_class_list_file(args.alt_class_list)

    class_col=trainer.get_preset("image_list_class_column")
    image_col=trainer.get_preset("image_list_file_column")

    if args.load_model: 

        del trainer.model_settings["freeze_layers"]

        trainer.load_model()
        trainer.read_class_list()
        trainer.read_image_list_file(class_col=class_col,image_col=image_col)
        trainer.image_list_apply_class_list()
        dataset.ask_retrain_note()
        dataset.set_model_trainer(trainer)
        dataset.open_dataset()

    else:

        trainer.copy_image_list_file()
        trainer.copy_class_list_file()

        trainer.read_image_list_file(class_col=class_col,image_col=image_col)
        trainer.read_class_list()

        trainer.image_list_apply_class_list()
        trainer.assemble_model()
        trainer.save_model_architecture()
        dataset.ask_note()
        dataset.set_model_trainer(trainer)
        dataset.make_dataset()

    dataset.update_model_state("training")
    dataset.save_dataset()

    # setting this only so they can be accessed in __del__, when someone Ctrl+C's the training
    trainer.set_dataset(dataset)
    trainer.set_timer(timer)

    trainer.configure_generators()
    trainer.train_model()

    dataset.set_epochs_trained(trainer.customcallback.get_current_epoch())
    dataset.set_training_time(timer.get_time_passed())
    dataset.update_model_state("skipped_training" if args.skip_training else "configured")

    dataset.save_dataset()

    trainer.save_model()
    trainer.save_history()
