#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:49:07 2019

@author: maarten
"""

  def cholletCode(self):
    self._setStaticModelCallbacks()
    self.logger.info("running chollet code")

    # https://github.com/keras-team/keras/blob/master/docs/templates/applications.md#fine-tune-inceptionv3-on-a-new-set-of-classes
    from keras.applications.inception_v3 import InceptionV3
#    from keras.preprocessing import image
    from keras.models import Model
    from keras.layers import Dense, GlobalAveragePooling2D
#    from keras import backend as K
    
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(self.train_generator.num_classes, activation='softmax')(x)

    # this is the model we will train
    self.model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    from keras.optimizers import RMSprop
    # optimizer=RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0),
    self.model.compile(optimizer=RMSprop(lr=0.0001), loss='categorical_crossentropy',
          metrics=["acc"])

    step_size_train=self.train_generator.n//self.train_generator.batch_size
    step_size_validate=self.validation_generator.n//self.validation_generator.batch_size
    
    # train the model on the new data for a few epochs
    self.history = self.model.fit_generator(
        self.train_generator,
        steps_per_epoch=step_size_train,
        epochs=4,
        validation_data=self.validation_generator,
        validation_steps=step_size_validate,
        callbacks=self.callbacks)
   
    self.logHistory()
    self.evaluateAndPredict()
   
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.
    
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    #let's not
    #for i, layer in enumerate(base_model.layers):
    #   print(i, layer.name)
    
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in self.model.layers[:249]:
       layer.trainable = False
    for layer in self.model.layers[249:]:
       layer.trainable = True
    
    # we need to recompile the model for thepandasse modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss="categorical_crossentropy",
          metrics=["acc"])
    
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    self.history = self.model.fit_generator(
        self.train_generator,
        steps_per_epoch=step_size_train,
        epochs=200,
        validation_data=self.validation_generator,
        validation_steps=step_size_validate,
        callbacks=self.callbacks)

    self.logHistory()
    self.evaluateAndPredict()
    self._saveModel()


