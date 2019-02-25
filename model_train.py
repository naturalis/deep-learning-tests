#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten
"""
import baseclass
import model_parameters, model_store
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
from sklearn.model_selection import StratifiedShuffleSplit

# https://datascience.stackexchange.com/questions/32814/error-while-using-flow-from-generator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class modelTrain(baseclass.baseClass):

  imageFolder = ""
  imageListFile = ""
  groupedClassList = None
  imageList = []
  staticCallbacks = []
  callbacks = []
  trainingSettings = {}
  useTestSplit = False
  imageSetFraction = None
  testdf = None

  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))

    self._setImageDataGeneratorSource()
    self._setOutputFilesAndFolders()


  def initializeTraining(self):
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
      self.trainingSettings["minimum_images_per_class"]=50

    self.logger.info("base settings: {}".format(self.getBaseSettings()))
    self.logger.info("training settings: {}".format(self.trainingSettings))
    
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
        image_data_generator = parameters["image_data_generator"],
        minimum_images_per_class = parameters["minimum_images_per_class"],
    )
    
  
  def _giveWarnings(self):
    if (self.trainingSettings["early_stopping"]==False or self.trainingSettings["early_stopping"]["use"]==False) and \
      self.trainingSettings["save"]["after_every_epoch"]==False:
      self.logger.warning("configured no early stopping or saving after every epoch, model won't be saved until all epochs are complete")
        
    if self.trainingSettings["save"]["after_every_epoch"]==False and self.trainingSettings["save"]["when_configured"]==False:
      self.logger.warning("configured no saving after every epoch and no saving when configured, model won't be saved!")
    

  def _setImageDataGeneratorSource(self):
    if self.getSetting('use_subdirs_as_classes')==True:
      self.imageDataGeneratorSource="directories"
    else:
      self.imageDataGeneratorSource="dataframes"
    self.logger.info("image generatior source: {}".format(self.imageDataGeneratorSource))


  def _setOutputFilesAndFolders(self):
    if self.imageDataGeneratorSource=="dataframes":
      if 'grouped_class_list' in self.getSetting('output_lists'):
        self.groupedClassList = os.path.join(self.projectRoot,self.getSetting('output_lists')['grouped_class_list'])


  def _readImageListFile(self):
    if not self.imageDataGeneratorSource=="directories":
      self.readImageListFile()
      self.logger.info("read infile {}; got {} records".format(self.imageListFile,len(self.traindf)))
    
    
  def _checkForDuplicateImages(self):
    if not self.imageDataGeneratorSource=="directories":
      if len(self.traindf.drop_duplicates(keep=False)) != len(self.traindf):
        raise ValueError("found duplicate images ({} unique out of {} total)"
                         .format(len(self.traindf.drop_duplicates(keep=False)),len(self.traindf)))


  def _applyImageMinimum(self):
    if self.trainingSettings["minimum_images_per_class"]==0:
      self.logger.info("no image minimum set")
      return

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
    if "test" in self.trainingSettings["split"]:
      self.useTestSplit=True
      sss = StratifiedShuffleSplit(n_splits=1, test_size=self.trainingSettings["split"]["test"], random_state=0)
  
      train_ids= []
      test_ids = []

      for train_index, test_index in sss.split(self.traindf["image"], self.traindf["label"]):
        train_ids.extend(train_index)
        test_ids.extend(test_index)
      
      self.testdf = self.traindf.ix[test_ids].reset_index()
      self.traindf = self.traindf.ix[train_ids].reset_index()
      self.logger.info("did {} test split: {} train, {} test".format(self.trainingSettings["split"]["test"],len(train_ids),len(test_ids)))
   
    else:
      self.useTestSplit=False
      self.logger.info("no test split")


  def _saveClassList(self,class_dict):
    with open(self.classListPath, 'w+') as file:
      c = csv.writer(file)
      c.writerow(["id","label"])
      for name, id in class_dict.items():
        c.writerow([id,name])

    self.logger.info("wrote {} classes to {}".format(len(class_dict),self.classListPath))

    if not self.imageDataGeneratorSource=="directories" and not self.groupedClassList == None:
      grouped_df=self.traindf.groupby(by=['label'])

      with open(self.groupedClassList, 'w+') as file:
        c = csv.writer(file)
        for key, item in grouped_df:
           c.writerow([key,len(grouped_df.get_group(key))])
#           print(grouped_df.get_group(key), "\n\n")
      self.logger.info("wrote class count list to {}".format(self.groupedClassList))


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

    self.logger.info("got {} classes".format(self.train_generator.num_classes))


  def logHistory(self):
    self.logger.info("val_loss: {}; acc: {}; loss: {}; val_acc: {}".format(
        self.history.history['val_loss'][0],
        self.history.history['acc'][0],
        self.history.history['loss'][0],
        self.history.history['val_acc'][0]))


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

      self.logHistory()

      # in case of fit_generator, the evaluation is part of the fitting itself
      scores = self.model.evaluate_generator(
          self.test_generator,
          self.test_generator.n/self.test_generator.batch_size,
          verbose=1)
  
     #      print(scores)
      print("test_generator")
      print(self.model.metrics_names[0], scores[0],self.model.metrics_names[1], scores[1])
  
      # in case of fit_generator, the evaluation is part of the fitting itself
      scores = self.model.evaluate_generator(
          self.validation_generator,
          self.validation_generator.n/self.validation_generator.batch_size,
          verbose=1)
      
      #      print(scores)
      print("validation_generator")
      print(self.model.metrics_names[0], scores[0],self.model.metrics_names[1], scores[1])


    from sklearn.metrics import classification_report, confusion_matrix
    #Confution Matrix and Classification Report
    Y_pred = self.model.predict_generator(self.validation_generator, self.validation_generator.n // self.validation_generator.batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)
    print('Confusion Matrix')
    print(confusion_matrix(self.validation_generator.classes, y_pred))
    print('Classification Report')
    print(classification_report(self.validation_generator.classes, y_pred))

    self._saveModel()






if __name__ == "__main__":
  start = time.time()

#  settings_file="./config/martin-collectie.yml"
#  settings_file="./config/aliens.yml"
#  settings_file="./config/mnist.yml"
  settings_file="./config/corvidae.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)
  params=model_parameters.modelParameters()
  store=model_store.modelStore()

  params.setModelParameters(
      model_name=settings["models"]["basename"]+"_150min",
      minimum_images_per_class=150,
      batch_size=32,
      split = { "validation": 0.2, "test" : 0.1  },
#      split = { "validation": 0.3 },
#      image_data_generator = {},
      early_stopping={ "use": True, "monitor": "val_acc", "patience": 3, "verbose": 0, "restore_best_weights": True},
      save={ "after_every_epoch": True, "after_every_epoch_monitor": "val_acc", "when_configured": True },
  )

  train=modelTrain(project_settings=settings, logger=logger)
  train.setModelVersionNumber(store.getVersionNumber())
  train.setModelParameters(params.getModelParameters())
  train.initializeTraining()
  train.trainModel()

  end=time.time()
  logger.info("{} took {}s".format(settings["project_name"],str(datetime.timedelta(seconds=end-start))))

