#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten.schermer@naturalis.nl
"""
import baseclass
import utilities
import model_parameters, model_store
import os, copy
import logging
from keras_preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras import applications
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import helpers.logger
import helpers.settings_reader
import time,datetime


# https://datascience.stackexchange.com/questions/32814/error-while-using-flow-from-generator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class modelTrain(baseclass.baseClass):

  imageFolder = ""
  imageListFile = ""
  imageList = []
  staticCallbacks = []
  callbacks = []
  useTestSplit = False
  imageSetFraction = None


  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))
    self.readClasses()


  def initializeTraining(self):
    self.readTrainAndTestImageLists()
    self._createGenerators()
    self._initModel()
    self._setStaticModelCallbacks()


  def setModelParameters(self,parameters):    
    self.setTrainingsettings(
        model_name = parameters["model_name"],
        early_stopping = parameters["early_stopping"],
        save = parameters["save"],
        model_architecture = parameters["model_architecture"],
        batch_size = parameters["batch_size"],
        split = parameters["split"],
        training_stages = parameters["training_stages"],
        model_target_size = parameters["model_target_size"],
        image_data_generator = parameters["image_data_generator"],
        minimum_images_per_class = parameters["minimum_images_per_class"],
    )
    
  
  def _giveWarnings(self):
#    print(self.classes)
#    print(self.classList)
    
    if (self.trainingSettings["early_stopping"]==False or self.trainingSettings["early_stopping"]["use"]==False) and \
      self.trainingSettings["save"]["after_every_epoch"]==False:
      self.logger.warning("configured no early stopping or saving after every epoch, model won't be saved until all epochs are complete")
        
    if self.trainingSettings["save"]["after_every_epoch"]==False and self.trainingSettings["save"]["when_configured"]==False:
      self.logger.warning("configured no saving after every epoch and no saving when configured, model won't be saved!")


  def _logHistory(self):
    self.logger.info("val_loss: {}; acc: {}; loss: {}; val_acc: {}".format(
        self.history.history['val_loss'][0],
        self.history.history['acc'][0],
        self.history.history['loss'][0],
        self.history.history['val_acc'][0]))


  def _saveModel(self):
    if self.trainingSettings["save"]["when_configured"]==True:
      self.model.save(self.modelFilePath)
      self.logger.info("saved model {}".format(self.modelFilePath))
    else:
      self.logger.info("skipped saving model")


  def _saveModelJson(self):
    f = open(self.modelJsonFilePath, "w")
    f.write(self.model.to_json())
    f.close()
    self.logger.info("saved model architecture to {}".format(self.modelJsonFilePath))


  def _setStaticModelCallbacks(self):
    if not self.trainingSettings["early_stopping"]==False:
      early_stopping = EarlyStopping(
          monitor=self.trainingSettings["early_stopping"]["monitor"],
          patience=self.trainingSettings["early_stopping"]["patience"],
          verbose=self.trainingSettings["early_stopping"]["verbose"],
          restore_best_weights=self.trainingSettings["early_stopping"]["restore_best_weights"]
          )
      self.staticCallbacks.append(early_stopping)
      self.logger.info("enabled early stopping ({})".format(self.trainingSettings["early_stopping"]["monitor"]))
    else:
      self.logger.info("no early stopping")

    if self.trainingSettings["save"]["after_every_epoch"]==True:
      checkpoint = ModelCheckpoint(
          filepath=self.modelFilePath,
          verbose=0, 
          save_best_only=True, 
          monitor=self.trainingSettings["save"]["after_every_epoch_monitor"], 
          mode='auto')
      self.staticCallbacks.append(checkpoint)
      self.logger.info("enabled saving after every epoch")
    else:
      self.logger.info("no saving after every epoch")


  def _setDynamicModelCallbacks(self,phase):
    self.callbacks=copy.deepcopy(self.staticCallbacks)

    if not phase["reduce_lr"]["use"]==False:
      reduce_lr = ReduceLROnPlateau(
          monitor=phase["reduce_lr"]["monitor"],
          factor=phase["reduce_lr"]["factor"],
          patience=phase["reduce_lr"]["patience"],
          verbose=0,
          min_lr=phase["reduce_lr"]["min_lr"],
          mode="auto")
      
      self.callbacks.append(reduce_lr)
      self.logger.info("enabled lr reduction ({})".format(phase["reduce_lr"]["monitor"]))      
    else:
      self.logger.info("no lr reduction")


  def _createGenerators(self):
    a = self.trainingSettings["image_data_generator"]

    # featurewise_center=False, 
    # samplewise_center=False, 
    # featurewise_std_normalization=Fals  e, 
    # samplewise_std_normalization=False, 
    # zca_whitening=False, 
    # zca_epsilon=1e-06, 
    # brightness_range=None, 
    # channel_shift_range=0.0, 
    # fill_mode='nearest', 
    # cval=0.0, 
    # data_format=None

    datagen=ImageDataGenerator(
        rotation_range=a["rotation_range"] if "rotation_range" in a else 0,
        shear_range=a["shear_range"] if "shear_range" in a else 0.0,
        zoom_range=a["zoom_range"] if "zoom_range" in a else 0.0,
        width_shift_range=a["width_shift_range"] if "width_shift_range" in a else 0.0,
        height_shift_range=a["height_shift_range"] if "height_shift_range" in a else 0.0,
        horizontal_flip=a["horizontal_flip"] if "horizontal_flip" in a else False,
        vertical_flip=a["vertical_flip"] if "vertical_flip" in a else False,
        preprocessing_function=self.preProcess,
        validation_split=self.trainingSettings["split"]["validation"])

    # https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
    self.train_generator=datagen.flow_from_dataframe(
        classes=self.classList,
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol, 
        class_mode="categorical", 
        target_size=self.modelTargetSize,
        batch_size=self.trainingSettings["batch_size"],
        interpolation="nearest",
        subset="training")
    
    self.validation_generator=datagen.flow_from_dataframe(
        classes=self.classList,
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol,
        class_mode="categorical", 
        target_size=self.modelTargetSize,
        batch_size=self.trainingSettings["batch_size"],
        interpolation="nearest",
        subset="validation")

    if self.useTestSplit==True:
      test_datagen=ImageDataGenerator(rescale=1./255.)
      
      self.test_generator=test_datagen.flow_from_dataframe(
          classes=self.classList,
          dataframe=self.testdf,
          directory=self.imageFolder,
          x_col=self.imageListImageCol, 
          y_col=self.imageListClassCol,
          batch_size=self.trainingSettings["batch_size"],
#          class_mode=None,
          class_mode="categorical", 
          target_size=self.modelTargetSize)

      self.logger.info("split: {} train, {} validation, {} test".format(
          self.train_generator.n,
          self.validation_generator.n,
          self.test_generator.n))

    else:      
      self.logger.info("split: {} train, {} validation (no test)".format(
          self.train_generator.n,
          self.validation_generator.n))

    self.logger.info("got {} classes".format(len(self.classList)))
    



  def _initModel(self):
    self.model = models.Sequential()

    # https://github.com/keras-team/keras/blob/master/docs/templates/applications.md#fine-tune-inceptionv3-on-a-new-set-of-classes
    # create the base pre-trained model     
    if self.modelArchitecture == "InceptionV3":
      self.conv_base = applications.InceptionV3(weights="imagenet",include_top=False)
    elif self.modelArchitecture == "VGG16":
      self.conv_base = applications.VGG16(weights="imagenet",include_top=False)
    else:
      raise ValueError("unsupported architecture {}".format(self.modelArchitecture))

    # add a global spatial average pooling layer
    x = self.conv_base.output
    x = layers.GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = layers.Dense(1024, activation='relu')(x)

    # and a logistic layer
    predictions = layers.Dense(len(self.classList), activation='softmax')(x)

    # this is the model we will train
    self.model=models.Model(inputs=self.conv_base.input, outputs=predictions)
    
    # saving the architecture
    self._saveModelJson()
   
    self.logger.info("model architecture: {}".format(self.modelArchitecture))
    self.logger.debug("number of layers: {}".format(len(self.model.layers)))


  def trainModel(self):
    self.logger.info("training model {}".format(self.modelName)) # no version?
          
    for idx, phase in enumerate(self.trainingPhases):
      if phase["use"]==False:
        self.logger.info("skipping phase {}".format(idx+1))
        continue

      # (un)freezing layers depending on training phase
      try: 
        int(phase["frozen_layers"])
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        #     https://github.com/keras-team/keras/issues/6910
        #     Yes, mixed8 is now the 248th layer.
        #     So it should be 249 instead of 172 in the doc.    
        for layer in self.model.layers[:phase["frozen_layers"]]:
           layer.trainable = False
        for layer in self.model.layers[phase["frozen_layers"]:]:
           layer.trainable = True

      except ValueError:
        if phase["frozen_layers"]=="base_model":
          # first: train only the top layers (which were randomly initialized)
          # i.e. freeze all convolutional InceptionV3 layers
          for layer in self.conv_base.layers:
            layer.trainable = False
        elif phase["frozen_layers"]=="none":
          # finally train them all
          for layer in self.model.layers:
            layer.trainable = True

      if idx==0 and self.logger.log_level==logging.DEBUG:
        self.model.summary()
        
      # compiling the model (should be done *after* setting layers to non-trainable)
      self.model.compile(
          optimizer=optimizers.RMSprop(lr=phase["initial_lr"]),
          loss=phase["loss_function"],
          metrics=["acc"]
      )

      # logging phase info
      trainable_count = int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
      non_trainable_count = int(np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)]))

      self.logger.info("phase {} of {} ({})".format(idx+1,len(self.trainingPhases),phase["label"]))
      self.logger.debug("parameters: {:,}; trainable: {:,}; non-trainable: {:,}".format(
          trainable_count + non_trainable_count,
          trainable_count,
          non_trainable_count)
      )

      # calculate step size and check validity
      step_size_train=self.train_generator.n//self.train_generator.batch_size
      step_size_validate=self.validation_generator.n//self.validation_generator.batch_size
    
      if step_size_train <= 0:
        raise ValueError("training step size 0 or less; reduce batch size to fix (currently {})".format(
            self.train_generator.batch_size))

      if step_size_validate <= 0:
        raise ValueError("validation step size 0 or less; reduce batch size to fix (currently {})".format(
            self.validation_generator.batch_size))

      self.logger.info("epochs: {} / steps_per_epoch: {} / validation_steps: {}".format(
          phase["epochs"],
          step_size_train,
          step_size_validate))
    
      # re-init callbacks (lr-reduction can differ per phase)
      self._setDynamicModelCallbacks(phase)    

      # fit the model
      self.history = self.model.fit_generator(
          self.train_generator,
          steps_per_epoch=step_size_train,
          epochs=phase["epochs"],
          validation_data=self.validation_generator,
          validation_steps=step_size_validate,
          callbacks=self.callbacks)

      self._logHistory()

    self._saveModel()



  def evaluateModel(self):
    # in case of fit_generator, the evaluation is part of the fitting itself
    print("evaluate: validation_generator")
    scores = self.model.evaluate_generator(
        self.validation_generator,
        self.validation_generator.n/self.validation_generator.batch_size,
        verbose=1)
    
    print(self.model.metrics_names[0], scores[0],self.model.metrics_names[1], scores[1])


    print("evaluate: test_generator")
    scores = self.model.evaluate_generator(
        self.test_generator,
        self.test_generator.n/self.test_generator.batch_size,
        verbose=1)

    print(self.model.metrics_names[0], scores[0],self.model.metrics_names[1], scores[1])





if __name__ == "__main__":
  start = time.time()

  settings_file=utilities.utilities.getSettingsFilePath()
  custom_params_file=utilities.utilities.getCustomParametersFilePath()

  settings = helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)
  store = model_store.modelStore(project_settings=settings, logger=logger)
  params = model_parameters.modelParameters(logger=logger)

  params.readCustomSettingsFile(custom_params_file)
  params.setModelParameters(model_name=settings["models"]["basename"])
 
  train=modelTrain(project_settings=settings, logger=logger)
  train.setModelVersionNumber(store.generateVersionNumber())
  train.setModelParameters(params.getModelParameters())
  train.initializeTraining()
  train.trainModel()
  train.evaluateModel()

  end=time.time()
  logger.info("{} took {}s".format(settings["project_name"],str(datetime.timedelta(seconds=end-start))))
