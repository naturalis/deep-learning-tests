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

        # x = tf.keras.layers.Dropout(.2)(x)
        # x = tf.keras.layers.Dense(1024, activation='relu')(x)

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
    trainer.set_project_folders(project_root=os.environ['PROJECT_ROOT'])
    trainer.set_model_name()
    trainer.set_model_folder()
    
    if 'CLASS_IMAGE_MINIMUM' in os.environ:
        trainer.set_class_image_minimum(os.environ['CLASS_IMAGE_MINIMUM'])

    # if 'CLASS_IMAGE_MAXIMUM' in os.environ:
    #     trainer.set_class_image_maximum(os.environ['CLASS_IMAGE_MAXIMUM'])

    trainer.copy_class_list_file()
    trainer.read_class_list()
    trainer.class_list_apply_image_minimum()
    trainer.read_image_list_file(image_col=2)
    trainer.image_list_apply_class_list()

        # "base_model": tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False),  
        # "base_model": tf.keras.applications.ResNet50(weights="imagenet", include_top=False),

        # WARNING:tensorflow:Method (on_train_batch_end) is slow compared to the batch update (0.325404). Check your callbacks.
        # maybe something with TensorBoard callback, as the other ones get called at epoch end, not batch end

        # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/LearningRateScheduler

    learning_rate = float(os.environ["INITIAL_LR"]) if "INITIAL_LR" in os.environ else 1e-4
    batch_size = int(os.environ["BATCH_SIZE"]) if "BATCH_SIZE" in os.environ else 64

    trainer.logger.info("learning_rate: {}".format(learning_rate))
    trainer.logger.info("batch_size: {}".format(batch_size))


    basemodel = tf.keras.applications.InceptionV3(weights="imagenet", include_top=False)
    # ['_TF_MODULE_IGNORED_PROPERTIES', '__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_activity_regularizer', '_add_inbound_node', '_add_unique_metric_name', '_add_variable_with_custom_getter', '_assert_compile_was_called', '_assert_weights_created', '_autocast', '_base_init', '_build_model_with_inputs', '_cache_output_metric_attributes', '_call_accepts_kwargs', '_call_arg_was_passed', '_call_fn_args', '_callable_losses', '_check_call_args', '_check_trainable_weights_consistency', '_checkpoint_dependencies', '_clear_losses', '_collect_input_masks', '_compile_distribution', '_compile_eagerly', '_compile_from_inputs', '_compile_time_distribution_strategy', '_compile_weights_loss_and_weighted_metrics', '_compute_dtype', '_compute_output_and_mask_jointly', '_dedup_weights', '_deferred_dependencies', '_distribution_standardize_user_data', '_distribution_strategy', '_dtype', '_dtype_defaulted_to_floatx', '_dtype_policy', '_dynamic', '_eager_add_metric', '_eager_losses', '_expects_mask_arg', '_expects_training_arg', '_experimental_run_tf_function', '_feed_input_names', '_feed_input_shapes', '_feed_inputs', '_feed_loss_fns', '_feed_output_names', '_feed_output_shapes', '_feed_sample_weights', '_feed_targets', '_flatten', '_gather_children_attribute', '_gather_saveables_for_checkpoint', '_get_call_arg_value', '_get_callback_model', '_get_existing_metric', '_get_node_attribute_at_index', '_get_trainable_state', '_get_training_eval_metrics', '_graph', '_graph_network_add_loss', '_graph_network_add_metric', '_handle_activity_regularization', '_handle_deferred_dependencies', '_handle_metrics', '_handle_per_output_metrics', '_handle_weight_regularization', '_inbound_nodes', '_init_call_fn_args', '_init_distributed_function_cache_if_not_compiled', '_init_graph_network', '_init_metric_attributes', '_init_set_name', '_init_subclassed_network', '_input_coordinates', '_input_layers', '_insert_layers', '_is_compiled', '_is_graph_network', '_is_layer', '_keras_api_names', '_keras_api_names_v1', '_layer_call_argspecs', '_layers', '_list_extra_dependencies_for_serialization', '_list_functions_for_serialization', '_lookup_dependency', '_loss_weights_list', '_losses', '_make_callback_model', '_make_execution_function', '_make_predict_function', '_make_test_function', '_make_train_function', '_maybe_build', '_maybe_cast_inputs', '_maybe_create_attribute', '_maybe_initialize_trackable', '_maybe_load_initial_epoch_from_ckpt', '_metrics', '_name', '_name_based_attribute_restore', '_name_based_restores', '_name_scope', '_nested_inputs', '_nested_outputs', '_network_nodes', '_no_dependency', '_nodes_by_depth', '_non_trainable_weights', '_obj_reference_counts', '_obj_reference_counts_dict', '_object_identifier', '_outbound_nodes', '_output_coordinates', '_output_layers', '_output_loss_metrics', '_output_mask_cache', '_output_shape_cache', '_output_tensor_cache', '_preload_simple_restoration', '_prepare_output_masks', '_prepare_sample_weights', '_prepare_skip_target_masks', '_prepare_total_loss', '_prepare_validation_data', '_process_target_tensor_for_compile', '_recompile_weights_loss_and_weighted_metrics', '_restore_from_checkpoint_position', '_reuse', '_run_eagerly', '_run_internal_graph', '_sample_weight_modes', '_scope', '_select_training_loop', '_self_name_based_restores', '_self_setattr_tracking', '_self_unconditional_checkpoint_dependencies', '_self_unconditional_deferred_dependencies', '_self_unconditional_dependency_names', '_self_update_uid', '_set_connectivity_metadata_', '_set_dtype_policy', '_set_input_attrs', '_set_inputs', '_set_mask_metadata', '_set_metric_attributes', '_set_optimizer', '_set_output_attrs', '_set_output_names', '_set_per_output_metric_attributes', '_set_trainable_state', '_setattr_tracking', '_should_compute_mask', '_single_restoration_from_checkpoint_position', '_standardize_user_data', '_symbolic_add_metric', '_symbolic_call', '_targets', '_tf_api_names', '_tf_api_names_v1', '_thread_local', '_track_layers', '_track_trackable', '_trackable_saved_model_saver', '_trackable_saver', '_tracking_metadata', '_trainable', '_trainable_weights', '_unconditional_checkpoint_dependencies', '_unconditional_dependency_names', '_undeduplicated_weights', '_update_sample_weight_modes', '_update_uid', '_updated_config', '_updates', '_validate_compile_param_for_distribution_strategy', '_validate_graph_inputs_and_outputs', '_validate_or_infer_batch_size', '_warn_about_input_casting', 'activity_regularizer', 'add_loss', 'add_metric', 'add_update', 'add_variable', 'add_weight', 'apply', 'build', 'built', 'call', 'compile', 'compute_mask', 'compute_output_shape', 'compute_output_signature', 'count_params', 'dtype', 'dynamic', 'evaluate', 'evaluate_generator', 'fit', 'fit_generator', 'from_config', 'get_config', 'get_input_at', 'get_input_mask_at', 'get_input_shape_at', 'get_layer', 'get_losses_for', 'get_output_at', 'get_output_mask_at', 'get_output_shape_at', 'get_updates_for', 'get_weights', 'inbound_nodes', 'input', 'input_mask', 'input_names', 'input_shape', 'input_spec', 'inputs', 'layers', 'load_weights', 'losses', 'metrics', 'metrics_names', 'name', 'name_scope', 'non_trainable_variables', 'non_trainable_weights', 'optimizer', 'outbound_nodes', 'output', 'output_mask', 'output_names', 'output_shape', 'outputs', 'predict', 'predict_generator', 'predict_on_batch', 'reset_metrics', 'reset_states', 'run_eagerly', 'sample_weights', 'save', 'save_weights', 'set_weights', 'state_updates', 'stateful', 'submodules', 'summary', 'supports_masking', 'test_on_batch', 'to_json', 'to_yaml', 'train_on_batch', 'trainable', 'trainable_variables', 'trainable_weights', 'updates', 'variables', 'weights', 'with_name_scope']

    off = tf.keras.losses.CategoricalCrossentropy()
    # ['__call__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_fn_kwargs', '_get_reduction', '_keras_api_names', '_keras_api_names_v1', 'call', 'fn', 'from_config', 'get_config', 'name', 'reduction']

    andd = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    # ['__abstractmethods__', '__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_abc_cache', '_abc_negative_cache', '_abc_negative_cache_version', '_abc_registry', '_add_variable_with_custom_getter', '_assert_valid_dtypes', '_call_if_callable', '_checkpoint_dependencies', '_compute_gradients', '_create_hypers', '_create_or_restore_slot_variable', '_create_slots', '_decayed_lr', '_deferred_dependencies', '_deferred_slot_restorations', '_dense_apply_args', '_distributed_apply', '_fallback_apply_state', '_gather_saveables_for_checkpoint', '_get_hyper', '_handle_deferred_dependencies', '_hyper', '_hypers_created', '_init_set_name', '_initial_decay', '_iterations', '_keras_api_names', '_keras_api_names_v1', '_list_extra_dependencies_for_serialization', '_list_functions_for_serialization', '_lookup_dependency', '_maybe_initialize_trackable', '_momentum', '_name', '_name_based_attribute_restore', '_name_based_restores', '_no_dependency', '_object_identifier', '_preload_simple_restoration', '_prepare', '_prepare_local', '_resource_apply_dense', '_resource_apply_sparse', '_resource_apply_sparse_duplicate_indices', '_resource_scatter_add', '_resource_scatter_update', '_restore_from_checkpoint_position', '_restore_slot_variable', '_serialize_hyperparameter', '_set_hyper', '_setattr_tracking', '_single_restoration_from_checkpoint_position', '_slot_names', '_slots', '_sparse_apply_args', '_track_trackable', '_tracking_metadata', '_unconditional_checkpoint_dependencies', '_unconditional_dependency_names', '_update_uid', '_use_locking', '_valid_dtypes', '_weights', 'add_slot', 'add_weight', 'apply_gradients', 'centered', 'epsilon', 'from_config', 'get_config', 'get_gradients', 'get_slot', 'get_slot_names', 'get_updates', 'get_weights', 'iterations', 'minimize', 'set_weights', 'variables', 'weights']

    die = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="auto", restore_best_weights=True, verbose=1)
    # ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_chief_worker_only', '_keras_api_names', '_keras_api_names_v1', 'baseline', 'best_weights', 'get_monitor_value', 'min_delta', 'model', 'monitor', 'monitor_op', 'on_batch_begin', 'on_batch_end', 'on_epoch_begin', 'on_epoch_end', 'on_predict_batch_begin', 'on_predict_batch_end', 'on_predict_begin', 'on_predict_end', 'on_test_batch_begin', 'on_test_batch_end', 'on_test_begin', 'on_test_end', 'on_train_batch_begin', 'on_train_batch_end', 'on_train_begin', 'on_train_end', 'patience', 'restore_best_weights', 'set_model', 'set_params', 'stopped_epoch', 'validation_data', 'verbose', 'wait']


    # print(dir(basemodel))
    print(basemodel.summary)
    print(basemodel.name)

    # print(dir(off))
    # print(dir(andd))
    # print(dir(die))

    exit(0)


    trainer.set_model_settings({
        "validation_split": 0.2,
        "base_model": tf.keras.applications.InceptionV3(weights="imagenet", include_top=False),
        "loss": tf.keras.losses.CategoricalCrossentropy(),
        "optimizer": [
            tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        ],
        "batch_size": batch_size,
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


    # env variables
    #     PROJECT_ROOT
    #     CLASS_IMAGE_MINIMUM=10 / 2
    #     CLASS_IMAGE_MAXIMUM=1000 / None

    # model params /  hyperparameters
    #     "validation_split": 0.2,
    #     "base_model": tf.keras.applications.InceptionV3(weights="imagenet", include_top=False),  
    #     "loss": tf.keras.losses.CategoricalCrossentropy(),
    #     "optimizer": [
    #         tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    #     ],
    #     "batch_size": batch_size,
    #     "epochs": [ 200 ], # epochs single value or list controls whether training is phased
    #     "freeze_layers": [ "none" ], # "base_model", # 249, # none
    #     "callbacks" : [
    #         [ 
    #             tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="auto", restore_best_weights=True, verbose=1),
    #             # tf.keras.callbacks.TensorBoard(trainer.get_tensorboard_log_path()),
    #             tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=4, min_lr=1e-8, verbose=1),
    #             tf.keras.callbacks.ModelCheckpoint(trainer.get_model_path(), monitor="val_acc", save_best_only=True, save_freq="epoch", verbose=1)
    #         ]
    #     ],
    #     "metrics" : [ "acc" ],
    #     "image_augmentation" : {
    #         "rotation_range": 90,
    #         "shear_range": 0.2,
    #         "zoom_range": 0.2,
    #         "horizontal_flip": True,
    #         "width_shift_range": 0.2,
    #         "height_shift_range": 0.2, 
    #         "vertical_flip": False
    #     }

    # list of images
    #     list of URLs!
    #     classes?

    # naam / notes