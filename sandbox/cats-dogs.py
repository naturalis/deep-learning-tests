from __future__ import absolute_import, division, print_function, unicode_literals

import os
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from lib import logclass


class ModelTrainer():

    IMG_SIZE = 160 # All images will be resized to 160x160
    BATCH_SIZE = 32
    SHUFFLE_BUFFER_SIZE = 1000

    def __init__(self):
        self.logger = logclass.LogClass(self.__class__.__name__)
        self.logger.info("TensorFlow v{}".format(tf.__version__))

    def format_example(self, image, label):
        image = tf.cast(image, tf.float32)
        image = (image/127.5) - 1
        image = tf.image.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        return image, label

    def train_example(self):

        IMG_SHAPE = (self.IMG_SIZE, self.IMG_SIZE, 3)

        import tensorflow_datasets as tfds
        tfds.disable_progress_bar()

        (raw_train, raw_validation, raw_test), metadata = tfds.load(
            'cats_vs_dogs',
            split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
            with_info=True,
            as_supervised=True,
        )

        train = raw_train.map(self.format_example)
        validation = raw_validation.map(self.format_example)
        test = raw_test.map(self.format_example)

        train_batches = train.shuffle(self.SHUFFLE_BUFFER_SIZE).batch(self.BATCH_SIZE)
        validation_batches = validation.batch(self.BATCH_SIZE)
        test_batches = test.batch(self.BATCH_SIZE)

        for image_batch, label_batch in train_batches.take(1):
           pass

        # Create the base model from the pre-trained model MobileNet V2
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

        feature_batch = base_model(image_batch)
        print(feature_batch.shape)

        base_model.trainable = False
        base_model.summary()

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        feature_batch_average = global_average_layer(feature_batch)
        print(feature_batch_average.shape)

        prediction_layer = tf.keras.layers.Dense(1)
        prediction_batch = prediction_layer(feature_batch_average)
        print(prediction_batch.shape)

        model = tf.keras.Sequential([
          base_model,
          global_average_layer,
          prediction_layer
        ])

        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        initial_epochs = 10
        validation_steps=20

        loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

        print("initial loss: {:.2f}".format(loss0))
        print("initial accuracy: {:.2f}".format(accuracy0))

        history = model.fit(train_batches,
                            epochs=initial_epochs,
                            validation_data=validation_batches)

        base_model.trainable = True

        # Let's take a look to see how many layers are in the base model
        print("Number of layers in the base model: ", len(base_model.layers))

        # Fine-tune from this layer onwards
        fine_tune_at = 100

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
                      metrics=['accuracy'])

        model.summary()

        fine_tune_epochs = 10
        total_epochs =  initial_epochs + fine_tune_epochs

        history_fine = model.fit(train_batches,
                                 epochs=total_epochs,
                                 initial_epoch =  history.epoch[-1],
                                 validation_data=validation_batches)

        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']

        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']


if __name__ == "__main__":

    trainer = ModelTrainer()
    trainer.train_example()
