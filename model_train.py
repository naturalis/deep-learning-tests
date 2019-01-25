#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten
"""
import baseclass
import os, csv 
import logging
from keras_preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import helpers.logger
import helpers.settings_reader
import numpy as np

# https://datascience.stackexchange.com/questions/32814/error-while-using-flow-from-generator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class modelTrain(baseclass.baseClass):

  project_dir = "./"
  imageFolder = ""
  imageListFile = ""
  imageList = []
  imageListClassCol = "label"
  imageListImageCol = "image"
  modelRepoFolder = "./"
  traind =""
  callbacks = []
  trainingSettings = {}

  def __init__(self,settings,logger=None):
    if logger is not None:
      self.logger = logger
    self.settings = settings
    self.setProjectRoot(settings)
    self.setModelRepoFolder(settings)
    self.setClassListPath(settings)
    self._readSettings()


  def setTrainingsettings(self, model_name, epochs, batch_size, model_architecture,
                          validation_split, early_stopping, save, initial_lr, reduce_lr, **kwargs):
    self.setModelName(model_name)
    self.setModelArchitecture(model_architecture)

    self.trainingSettings["batch_size"]=batch_size
    self.trainingSettings["epochs"]=epochs
    self.trainingSettings["validation_split"]=validation_split
    self.trainingSettings["initial_lr"]=initial_lr
    self.trainingSettings["save"]=save

    if not early_stopping==False and not early_stopping["use"]==False:
        self.trainingSettings["early_stopping"]=early_stopping
    else:
        self.trainingSettings["early_stopping"]=False
    
    if not reduce_lr==False and not reduce_lr["use"]==False:
        self.trainingSettings["reduce_lr"]=reduce_lr
    else:
        self.trainingSettings["reduce_lr"]=False
        
    if 'model' in kwargs:
      self.trainingSettings["model"]=kwargs['model']

    if 'train_conv_base' in kwargs:
      self.trainingSettings["train_conv_base"]=kwargs['train_conv_base']

    if 'add_extra_toppings' in kwargs:
      self.trainingSettings["add_extra_toppings"]=kwargs['add_extra_toppings']

    if 'input_size' in kwargs:
      self.setModelInputSize(kwargs['input_size'])
    else:
      if self.modelArchitecture == "InceptionV3" or self.modelArchitecture == "Xception":
        self.setModelInputSize((299,299,3))
      elif self.modelArchitecture == "VGG16":
        self.setModelInputSize((224,224,3))
      else:
        raise ValueError("no input size set")

    self.logger.debug("base settings: {}".format(self.getBaseSettings()))
    self.logger.debug("training settings: {}".format(self.trainingSettings))


  def _readSettings(self):
    self.imageFolder = os.path.join(self.projectRoot,self.settings['image_download']['folder'])
    self.imageListFile = os.path.join(self.projectRoot,self.settings['output_lists']['downloaded_list'])
    self.imageListFileEncoding = self.settings['output_lists']['image_list']['encoding']
    self.imageListClassCol = self.settings['output_lists']['image_list']['col_class']
    self.imageListImageCol = self.settings['output_lists']['image_list']['col_image']


  def _createModelFolder(self):
    if not os.path.exists(self.modelRepoFolder):
      os.makedirs(self.modelRepoFolder)
      self.logger.debug("created model folder: {}".format(self.modelRepoFolder))
   

  def _readImageListFile(self):
    self.traindf=pd.read_csv(self.imageListFile,encoding=self.imageListFileEncoding)
    self.logger.info("read infile {}".format(self.imageListFile))


  def _setNumberOfClasses(self):
    # print(self.traindf)
    self.numberOfClasses = len(self.traindf.groupby('label').nunique())
    self.logger.info("found {} classes".format(self.numberOfClasses))


  def _saveClassList(self,class_dict):
    with open(self.classListPath, 'w+') as file:
      c = csv.writer(file)
      c.writerow(["id","label"])
      for name, id in class_dict.items():
        c.writerow([id,name])

    self.logger.info("wrote classes to {} ({})".format(self.classListPath, len(class_dict)))


  def _setModelCallbacks(self):
    if not self.trainingSettings["early_stopping"]==False:
      early_stopping = EarlyStopping(monitor=self.trainingSettings["early_stopping"]["monitor"], 
                                     patience=self.trainingSettings["early_stopping"]["patience"],
                                     verbose=self.trainingSettings["early_stopping"]["verbose"],
                                     restore_best_weights=self.trainingSettings["early_stopping"]["restore_best_weights"])
      self.callbacks.append(early_stopping)
      self.logger.info("enabled early stopping ({})".format(self.trainingSettings["early_stopping"]["monitor"]))
    else:
      self.logger.info("no early stopping")

    if self.trainingSettings["save"]["after_every_epoch"]==True:
      checkpoint = ModelCheckpoint(filepath=self.modelFilePath,
                                   verbose=1, 
                                   save_best_only=True, 
                                   monitor=self.trainingSettings["save"]["after_every_epoch_monitor"], 
                                   mode='auto')
      self.callbacks.append(checkpoint)
      self.logger.info("enabled saving after every epoch")
    else:
      self.logger.info("no saving after every epoch")

    if not self.trainingSettings["reduce_lr"] == False:
      reduce_lr = ReduceLROnPlateau(monitor=self.trainingSettings["reduce_lr"]["monitor"],
                                    factor=self.trainingSettings["reduce_lr"]["factor"],
                                    patience=self.trainingSettings["reduce_lr"]["patience"],
                                    verbose=1,
                                    mode="auto")
      self.callbacks.append(reduce_lr)
      self.logger.info("enabled lr reduction ({})".format(self.trainingSettings["reduce_lr"]["monitor"]))
    else:
      self.logger.info("no lr reduction")


  def _saveModel(self):
    if self.trainingSettings["save"]["when_configured"]==True:
      self.model.save(self.modelFilePath)
      self.logger.info("saved model {}".format(self.modelFilePath))
    else:
      self.logger.info("skipped saving model")


  def _initModel(self):
    self.model = models.Sequential()

    if self.modelArchitecture=='custom':
      self.model = self.trainingSettings["model"]
      self.model.add(layers.Dense(self.numberOfClasses, activation='softmax'))
    
    else:
      if self.modelArchitecture == "InceptionV3":
        from keras.applications import InceptionV3
        conv_base = InceptionV3(weights="imagenet",
                          include_top=False,
                          input_shape=self.modelInputSize)

      elif self.modelArchitecture == "Xception":
        from keras.applications import Xception
        conv_base = Xception(weights="imagenet",
                          include_top=False,
                          input_shape=self.modelInputSize)

      elif self.modelArchitecture == "VGG16":
        from keras.applications import VGG16
        conv_base = VGG16(weights="imagenet",
                          include_top=False,
                          input_shape=self.modelInputSize)

      else:
        raise ValueError("unknown model architecture {}".format(self.modelArchitecture))

      if "train_conv_base" in self.trainingSettings and self.trainingSettings["train_conv_base"]==True:
        conv_base.trainable = True
      else:
        conv_base.trainable = False

      self.model.add(conv_base)

      if "add_extra_toppings" in self.trainingSettings and self.trainingSettings["add_extra_toppings"]==True:
        self.model.add(layers.Conv2D(256, activation='relu', kernel_size=3))
        self.model.add(layers.MaxPooling2D((2,2)))

      self.model.add(layers.Flatten())
      self.model.add(layers.Dense(256, activation='relu'))
      self.model.add(layers.Dense(self.numberOfClasses, activation='softmax'))      
      
    self.logger.info("model architecture: {}".format(self.modelArchitecture))      

    self.model.compile(loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=self.trainingSettings["initial_lr"]),
        metrics=['acc'])

    self.model.summary()


  def trainModel(self):
    self._createModelFolder()
    self._readImageListFile()
    self._setNumberOfClasses()
  
    # randomizing the input
    self.traindf = self.traindf.sample(frac=1)
    
    datagen=ImageDataGenerator(
        rotation_range=90,
        shear_range=0.2,
        zoom_range=0.2,
#        width_shift_range=0.2, 
#        height_shift_range=0.2,
        horizontal_flip=True, 
#        vertical_flip=True,
        preprocessing_function=self.preProcess,
        validation_split=self.trainingSettings["validation_split"])
   
    # https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
    train_generator=datagen.flow_from_dataframe(
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol, 
        class_mode="categorical", 
        target_size=self.modelTargetSize,
        batch_size=self.trainingSettings["batch_size"],
        interpolation="nearest",
        subset="training")

    validation_generator=datagen.flow_from_dataframe(
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol, 
        class_mode="categorical", 
        target_size=self.modelTargetSize,
        batch_size=self.trainingSettings["batch_size"],
        interpolation="nearest",
        subset="validation")

#    print(self.traindf.groupby(by="label").count().sort_values(by="image",ascending=False))
#    
#    bla = validation_generator.class_indices
#    index_to_class = dict(zip(bla.values(),bla.keys()))
#    print(index_to_class[222])
 
       
#    test_datagen=ImageDataGenerator(rescale=1./255.)
#    
#    test_generator=test_datagen.flow_from_dataframe(
#        dataframe=testdf,
#        directory="./test/",
#        x_col="id",
#        y_col=None,epochs
#        batch_size=32,
#        seed=42,
#        shuffle=False,
#        class_mode=None,
#        target_size=(32,32))


    self._saveClassList(train_generator.class_indices)
    self.logger.info("training model {}".format(self.modelName))

    self._initModel()
    self._setModelCallbacks()

    step_size_train=train_generator.n//train_generator.batch_size
    step_size_validate=validation_generator.n//validation_generator.batch_size
    
    if step_size_train <= 0:
      raise ValueError("training step size 0 or less; reduce batch size to fix (currently {})".format(train_generator.batch_size))

    if step_size_validate <= 0:
      raise ValueError("validation step size 0 or less; reduce batch size to fix (currently {})".format(validation_generator.batch_size))

    self.logger.info("epochs: {} / steps_per_epoch: {} / validation_steps: {}".format(
        self.trainingSettings["epochs"],
        step_size_train,
        step_size_validate))
    
    history = self.model.fit_generator(
        train_generator,
        steps_per_epoch=step_size_train,
        epochs=self.trainingSettings["epochs"],
        validation_data=validation_generator,
        validation_steps=step_size_validate,
        callbacks=self.callbacks)

#    print(history)

    self._saveModel()
   
    #    should be done on test, or on validation if no test exists
    score = self.model.evaluate_generator(
        validation_generator,
        validation_generator.n/validation_generator.batch_size,
        verbose=1)

    self.logger.info("score: {}".format(set(zip(self.model.metrics_names,score))))

    predictions = self.model.predict_generator(
        validation_generator,
        validation_generator.n/validation_generator.batch_size,
        verbose=1)
    
    bla = np.argmax(predictions,axis=1)
    print(bla.shape)
    print(bla)
#    print(predictions)
#    print(type(predictions))
    
#    p = self.model.predict()
    



if __name__ == "__main__":
  #  settings_file = "./config/martin-collectie.yml"
  settings_file = "./config/aliens.yml"
  settings = helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger('./log/','training',logging.DEBUG)

#  import helpers.custom_models
#  custom = helpers.custom_models.customModels()
#  custom_model = custom.smallSequentialModel(input_shape=(299, 299, 3))

  train = modelTrain(settings, logger)
  train.setTrainingsettings(
        model_name="alien_predator_inception_2",
        model_architecture="InceptionV3", # InceptionV3, Xception, VGG16, custom
#        model=custom_model,
#        add_extra_toppings=True, 
        batch_size=32,
        epochs=100,
        initial_lr=1e-5,
        reduce_lr={"use": True,
                   "monitor": "val_loss",
                   "factor": 0.2,
                   "patience": 3},
        early_stopping={"use": True,
                        "monitor": "val_loss",
                        "patience": 5,
                        "verbose": 1,
                        "restore_best_weights": True},
        validation_split=0.1,
        save={"after_every_epoch": True,
              "after_every_epoch_monitor": "val_loss",
              "when_configured": True}
      )

  train.trainModel()

   
    
# make threeway split
# switchable hyperparameters
# save w/ hyperparmeters and ... (dataset)





