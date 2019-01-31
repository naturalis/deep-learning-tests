#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten
"""
import baseclass
import model_parameters
import os, csv, copy
import logging
from keras_preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
from keras import applications
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
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
  imageListClassCol = "label"
  imageListImageCol = "image"
  staticCallbacks = []
  trainingSettings = {}
  useTestSplit = False

  def __init__(self,settings,logger=None):
    if logger is not None:
      self.logger = logger
    self.settings = settings
    self.setProjectRoot(settings)
    self.setModelRepoFolder(settings)
    self.setClassListPath(settings)
    self._setImageDataGeneratorSource()
    self._setOutputFilesAndFolders()


  def initializeTraining(self):
    self._createModelFolder()
    self._readImageListFile()
    self._checkForDuplicateImages()
    self._applyImageMinimum()
    self._doTestSplit()
    self._createGenerators()
    self._saveClassList(self.train_generator.class_indices)
    self._initModel()
    self._setStaticModelCallbacks()


  def setTrainingsettings(self, model_name, batch_size, model_architecture, model_target_size,
                          split, early_stopping, save, training_stages, image_data_generator, **kwargs):
    self.setModelName(model_name)
    self.setModelArchitecture(model_architecture)
    self.setModelTargetSize(model_target_size)
    self.trainingSettings["batch_size"]=batch_size
    self.trainingSettings["split"]=split
    self.trainingSettings["save"]=save
    self.trainingSettings["image_data_generator"]=image_data_generator
    self.trainingPhases=training_stages

    if not early_stopping==False and not early_stopping["use"]==False:
        self.trainingSettings["early_stopping"]=early_stopping
    else:
        self.trainingSettings["early_stopping"]=False
    
    if 'minimum_images_per_class' in kwargs:
      self.trainingSettings["minimum_images_per_class"]=kwargs['minimum_images_per_class']
    else:
      self.trainingSettings["minimum_images_per_class"]=2

    self.logger.debug("base settings: {}".format(self.getBaseSettings()))
    self.logger.debug("training settings: {}".format(self.trainingSettings))
    
    self._giveWarnings()


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
        image_data_generator = parameters["image_data_generator"]
    )


  def _giveWarnings(self):
    if (self.trainingSettings["early_stopping"]==False or self.trainingSettings["early_stopping"]["use"]==False) and \
      self.trainingSettings["save"]["after_every_epoch"]==False:
      self.logger.warning("configured no early stopping or saving after every epoch, model won't be saved until all epochs are complete")
        
    if self.trainingSettings["save"]["after_every_epoch"]==False and self.trainingSettings["save"]["when_configured"]==False:
      self.logger.warning("configured no saving after every epoch and no saving when configured, model won't be saved!")
    

  def _setImageDataGeneratorSource(self):
    if 'use_subdirs_as_classes' in self.settings:
      self.imageDataGeneratorSource="directories" if self.settings['use_subdirs_as_classes']==True else "dataframes"
    else:
      self.imageDataGeneratorSource="dataframes"


  def _setOutputFilesAndFolders(self):
    self.imageFolder = os.path.join(self.projectRoot,self.settings['image_download']['folder'])
    if self.imageDataGeneratorSource=="dataframes":
      self.imageListFile = os.path.join(self.projectRoot,self.settings['output_lists']['downloaded_list'])
      self.imageListFileEncoding = self.settings['output_lists']['image_list']['encoding']
      self.imageListClassCol = self.settings['output_lists']['image_list']['col_class']
      self.imageListImageCol = self.settings['output_lists']['image_list']['col_image']


  def _createModelFolder(self):
    if not os.path.exists(self.modelRepoFolder):
      os.makedirs(self.modelRepoFolder)
      self.logger.debug("created model folder: {}".format(self.modelRepoFolder))
   

  def _readImageListFile(self):
    if not self.imageDataGeneratorSource=="directories":
      self.traindf=pd.read_csv(self.imageListFile,encoding=self.imageListFileEncoding)
      self.logger.info("read infile {}; got {} records".format(self.imageListFile,len(self.traindf)))
    
    
  def _checkForDuplicateImages(self):
    if not self.imageDataGeneratorSource=="directories":
      if len(self.traindf.drop_duplicates(keep=False)) != len(self.traindf):
        raise ValueError("found duplicate images ({} unique out of {} total)"
                         .format(len(self.traindf.drop_duplicates(keep=False)),len(self.traindf)))


  def _applyImageMinimum(self):
    if not self.imageDataGeneratorSource=="directories":
      grouped_df=self.traindf.groupby(by=['label'])
  
      labels_to_keep = []
      for key, item in grouped_df:
        # print(key,len(grouped_df.get_group(key)), "\n\n")
        # print(grouped_df.get_group(key), "\n\n")
        if len(grouped_df.get_group(key)) >= self.trainingSettings["minimum_images_per_class"]:
          labels_to_keep.append(key)
        
      self.logger.info("{} of {} classes do not make the {} image minimum".format(
          len(grouped_df)-len(labels_to_keep),
          len(grouped_df),
          self.trainingSettings["minimum_images_per_class"]))
      
      self.traindf=self.traindf[self.traindf['label'].isin(labels_to_keep)]
    else:
      self.logger.warning("function _applyImageMinimum() not implemented for flow_from_directory")


  def _doTestSplit(self):
    return
    # code is wrong, testdata should be split off class-balanced
    if "test" in self.trainingSettings["split"]:
      n = int(self.trainingSettings["split"]["test"] * len(self.traindf))
      self.testdf=self.traindf.copy().iloc[:n]
      self.traindf = self.traindf.iloc[n:].reset_index()
      self.useTestSplit=True
    else:
      self.useTestSplit=False

  def _saveClassList(self,class_dict):
    with open(self.classListPath, 'w+') as file:
      c = csv.writer(file)
      c.writerow(["id","label"])
      for name, id in class_dict.items():
        c.writerow([id,name])

    self.logger.info("wrote {} classes to {}".format(len(class_dict),self.classListPath))


  def _saveModel(self):
    if self.trainingSettings["save"]["when_configured"]==True:
      self.model.save(self.modelFilePath)
      self.logger.info("saved model {}".format(self.modelFilePath))
    else:
      self.logger.info("skipped saving model")


  def _setStaticModelCallbacks(self):
    if not self.trainingSettings["early_stopping"]==False:
      early_stopping = EarlyStopping(
          monitor=self.trainingSettings["early_stopping"]["monitor"],
          patience=self.trainingSettings["early_stopping"]["patience"],
          verbose=self.trainingSettings["early_stopping"]["verbose"],
          restore_best_weights=self.trainingSettings["early_stopping"]["restore_best_weights"])
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
    # featurewise_std_normalization=False, 
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

    if self.imageDataGeneratorSource=="directories":
      self.train_generator=datagen.flow_from_directory(
          directory=self.imageFolder,
          class_mode="categorical", 
          target_size=self.modelTargetSize,
          batch_size=self.trainingSettings["batch_size"],
          interpolation="nearest",
          subset="training")

      self.validation_generator=datagen.flow_from_directory(
          directory=self.imageFolder,
          class_mode="categorical", 
          target_size=self.modelTargetSize,
          batch_size=self.trainingSettings["batch_size"],
          interpolation="nearest",
          subset="validation")
      
    else:
      # randomizing the input
      # self.traindf = self.traindf.sample(frac=1)

      # https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
      self.train_generator=datagen.flow_from_dataframe(
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
          dataframe=self.testdf,
          directory=self.imageFolder,
          x_col=self.imageListImageCol, 
          y_col=None,
          batch_size=self.trainingSettings["batch_size"],
          class_mode=None,
          target_size=self.modelTargetSize)

      self.logger.info("split: {} train, {} validation, {} test".format(
          self.train_generator.n,
          self.validation_generator.n,
          self.test_generator.n))

    else:      
      self.logger.info("split: {} train, {} validation (no test)".format(
          self.train_generator.n,
          self.validation_generator.n))

    self.logger.info("got {} classes".format(self.train_generator.num_classes))


  def evaluateAndPredict(self):
    if self.useTestSplit==True:
      eval_gen = self.test_generator
    else:
      eval_gen = self.validation_generator

    #  should be done on test, or on validation if no test exists
    score = self.model.evaluate_generator(
        eval_gen,
        eval_gen.n/eval_gen.batch_size,
        verbose=1)

    self.logger.info("score: {}".format(set(zip(self.model.metrics_names,score))))

#    predictions = self.model.predict_generator(
#        self.validation_generator,
#        self.validation_generator.n/self.validation_generator.batch_size,
#        verbose=1)
#
#    pred_argmax = np.argmax(predictions,axis=1)
#    print(pred_argmax.shape)
#    print(pred_argmax)
#    
#    self.logger.info("pred_argmax: {}".format(pred_argmax))
#    self.logger.info("pred_argmax.shape: {}".format(pred_argmax.shape))
    
#    p = self.model.predict()


  def _initModel(self):
    self.model = models.Sequential()

    #  https://github.com/keras-team/keras/blob/master/docs/templates/applications.md#fine-tune-inceptionv3-on-a-new-set-of-classes
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
    # x = layers.Dropout(0.2)(x) # not in chollet, but added in some inception schematics

    # let's add a fully-connected layer
    x = layers.Dense(1024, activation='relu')(x)

    # and a logistic layer
    predictions = layers.Dense(self.train_generator.num_classes, activation='softmax')(x)

    # this is the model we will train
    self.model=models.Model(inputs=self.conv_base.input, outputs=predictions)
   
    self.logger.info("model architecture: {}".format(self.modelArchitecture))     
    self.logger.debug("number of layers: {}".format(len(self.model.layers)))


  def trainModel(self):    
    self.logger.info("training model {}".format(self.modelName))
          
    for idx, phase in enumerate(self.trainingPhases):
      if phase["use"]==False:
        self.logger.info("skipping phase {}".format(idx+1))
        continue

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

      # compile the model (should be done *after* setting layers to non-trainable)
      self.model.compile(
          optimizer=optimizers.RMSprop(lr=phase["initial_lr"]),
          loss=phase["loss_function"],
          metrics=["acc"]
      )

      trainable_count = int(np.sum([K.count_params(p) for p in set(self.model.trainable_weights)]))
      non_trainable_count = int(np.sum([K.count_params(p) for p in set(self.model.non_trainable_weights)]))

      self.logger.info("phase {} of {} ({})".format(idx+1,len(self.trainingPhases),phase["label"]))
      self.logger.debug("parameters: {:,}; trainable: {:,}; non-trainable: {:,}".format(
          trainable_count + non_trainable_count,
          trainable_count,
          non_trainable_count)
      )

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
    
      self._setDynamicModelCallbacks(phase)    

      history = self.model.fit_generator(
          self.train_generator,
          steps_per_epoch=step_size_train,
          epochs=phase["epochs"],
          validation_data=self.validation_generator,
          validation_steps=step_size_validate,
          callbacks=self.callbacks)

      # print(history) save somehow
  
    self._saveModel()


  def cholletCode(self):
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
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
          metrics=["acc"])

    step_size_train=self.train_generator.n//self.train_generator.batch_size
    step_size_validate=self.validation_generator.n//self.validation_generator.batch_size
    
    # train the model on the new data for a few epochs
    model.fit_generator(
        self.train_generator,
        steps_per_epoch=step_size_train,
        epochs=4,
        validation_data=self.validation_generator,
        validation_steps=step_size_validate)
    
    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.
    
    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)
    
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True
    
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    from keras.optimizers import SGD
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
          metrics=["acc"])
    
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit_generator(
        self.train_generator,
        steps_per_epoch=step_size_train,
        epochs=400,
        validation_data=self.validation_generator,
        validation_steps=step_size_validate)

    self._saveModel()


if __name__ == "__main__":
  start = time.time()

  settings_file="./config/martin-collectie.yml"
#  settings_file="./config/aliens.yml"
#  settings_file="./config/mnist.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.DEBUG)
  params=model_parameters.modelParameters()

  params.setModelParameters(
      model_name=settings["models"]["basename"],
      minimum_images_per_class=10,
      # early_stopping={ "use": True, "monitor": "val_acc", "patience": 3, "verbose": 0, "restore_best_weights": True},
      # save={ "after_every_epoch": True, "after_every_epoch_monitor": "val_acc", "when_configured": True },
  )

  train=modelTrain(settings, logger)
  train.setModelParameters(params.getModelParameters())
  train.initializeTraining()
#  train.trainModel()
  train.cholletCode()

  end=time.time()
  logger.info("{} took {}s".format(settings["project_name"],str(datetime.timedelta(seconds=end-start))))
  
#  train.evaluateAndPredict()

