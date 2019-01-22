#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c

@author: maarten
"""
import os, csv, collections 
import logging
from keras_preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import pandas as pd
from keras.applications import InceptionV3, Xception
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import helpers.logger
import helpers.settings_reader
import numpy as np
import matplotlib.pyplot as plt

# https://datascience.stackexchange.com/questions/32814/error-while-using-flow-from-generator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class modelTrainer:
  settings = {}
  project_dir = "./"
  imageFolder = ""
  imageListFile = ""
  imageList = []
  imageListClassCol = "label"
  imageListImageCol = "image"
  modelRepoFolder = "./"
  classDictionaryFile = "classes.csv"
  traind =""
  callbacks = []
  trainingSettings = {}

  def __init__(self,settings,logger=None):
    self.settings = settings
    if logger is not None:
      self.logger = logger
    self._readSettings()


  def setTrainingsettings(self, 
                          model_name,
                          epochs,
                          batch_size,
                          model_architecture,
                          input_size,
                          validation_split,
                          early_stopping,
                          save,
                          initial_lr,
                          reduce_lr,
                          **kwargs):
    self.trainingSettings["model_name"]=model_name
    self.trainingSettings["epochs"]=epochs
    self.trainingSettings["batch_size"]=batch_size
    self.trainingSettings["model_architecture"]=model_architecture
    self.trainingSettings["input_size"]=input_size
    self.trainingSettings["target_size"]=self.trainingSettings["input_size"][0:2]
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
        
    self.modelSavePath = os.path.join(self.modelRepoFolder,self.trainingSettings["model_name"] + '.hd5')

    if 'model' in kwargs:
      self.trainingSettings["model"]=kwargs['model']

    self.logger.debug("training settings: {}".format(self.trainingSettings))


  def _readSettings(self):
    self.project_root = self.settings['project_root']
    self.imageFolder = os.path.join(self.project_root,self.settings['image_download']['folder'])
    self.classDictionaryFile = os.path.join(self.project_root,self.settings['output_lists']['class_list'])
    self.imageListFile = os.path.join(self.project_root,self.settings['output_lists']['downloaded_list'])
    self.imageListFileEncoding = self.settings['output_lists']['image_list']['encoding']
    self.imageListClassCol = self.settings['output_lists']['image_list']['col_class']
    self.imageListImageCol = self.settings['output_lists']['image_list']['col_image']
    self.modelRepoFolder = os.path.join(self.project_root,self.settings['models']['folder'])


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
    with open(self.classDictionaryFile, 'w+') as file:
      c = csv.writer(file)
      c.writerow(["id","label"])
      for name, id in class_dict.items():
        c.writerow([id,name])

    self.logger.info("wrote classes to {} ({})".format(self.classDictionaryFile, len(class_dict)))


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
      checkpoint = ModelCheckpoint(filepath=self.modelSavePath, 
                                   verbose=1, 
                                   save_best_only=True, 
                                   monitor=self.trainingSettings["save"]["monitor"], 
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
      self.model.save(self.modelSavePath)
      self.logger.info("saved model {}".format(self.modelSavePath))
    else:
      self.logger.info("skipped saving model")


  def _initModel(self):
    self.model = models.Sequential()

    if self.trainingSettings["model_architecture"]=='custom':
      self.model = self.trainingSettings["model"]
      self.model.add(layers.Dense(self.numberOfClasses, activation='softmax'))
    
    else:
      if self.trainingSettings["model_architecture"] == 'Inception':
        conv_base = InceptionV3(weights='imagenet',
                          include_top=False,
                          input_shape=self.trainingSettings["input_size"])
      elif self.trainingSettings["model_architecture"] == 'Xception':
        conv_base = Xception(weights='imagenet',
                          include_top=False,
                          input_shape=self.trainingSettings["input_size"])
      else:
        raise ValueError('missing model architecture')
      conv_base.trainable = False
  
      self.model.add(conv_base)
      self.model.add(layers.Flatten())
      self.model.add(layers.Dense(256, activation='relu'))
      self.model.add(layers.Dense(self.numberOfClasses, activation='softmax'))      
      
    self.logger.info("model architecture: {}".format(self.trainingSettings["model_architecture"]))      

    self.model.compile(loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(lr=self.trainingSettings["initial_lr"]),
        metrics=['acc'])

    self.model.summary()



  def trainModel(self):
    self._createModelFolder()
    self._readImageListFile()
    self._setNumberOfClasses()
  
    # randomizing the input
    # self.traindf = self.traindf.sample(frac=1)
  
    datagen=ImageDataGenerator(rescale=1./255.,validation_split=self.trainingSettings["validation_split"])

   
    train_generator=datagen.flow_from_dataframe(
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol, 
        class_mode="categorical", 
        target_size=self.trainingSettings["target_size"],
        batch_size=self.trainingSettings["batch_size"],
        interpolation="nearest",
        subset="training")

    validation_generator=datagen.flow_from_dataframe(
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol, 
        class_mode="categorical", 
        target_size=self.trainingSettings["target_size"],
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
    self.logger.info("training model {}".format(self.trainingSettings["model_name"]))

    self._initModel()
    self._setModelCallbacks()

    step_size_train=train_generator.n//train_generator.batch_size
    step_size_validate=validation_generator.n//validation_generator.batch_size

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
  train = modelTrainer(settings, logger)
  train.setTrainingsettings(
        model_name="alien_predator_1",
        model_architecture="Inception", # Inception, Xception
        batch_size=64,
        epochs=100,
        initial_lr=1e-5,
        reduce_lr={"use": True,
                   "monitor": "val_loss",
                   "factor": 0.2,
                   "patience": 3},
        early_stopping={"use": False,
                        "monitor": "val_loss",
                        "patience": 2,
                        "verbose": 1,
                        "restore_best_weights": True},
        input_size=(299, 299, 3),
        validation_split=0.1,
        save={"after_every_epoch": True,
              "monitor": "val_loss",
              "when_configured": True}
      )
  train.trainModel()



# make threeway split
# switchable hyperparameters
# save w/ hyperparmeters and ... (dataset)

# outputs:
#  confusion matrix
#  list of all test images + their predictions
