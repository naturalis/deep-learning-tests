#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten.schermer@naturalis.nl
"""
import baseclass
import utilities
import model_parameters
import logging, os, sys
import numpy as np
import pandas as pd
import helpers.logger
import helpers.settings_reader
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import decode_predictions, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# https://datascience.stackexchange.com/questions/32814/error-while-using-flow-from-generator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class modelTest(baseclass.baseClass):
  
  testdf = None
  ran_predictions = False
  
  def __init__(self, project_settings, logger):
    self.logger = logger
    self.setSettings(project_settings)


  def initTesting(self):
    self.readTrainAndTestImageLists()
    self.readClasses()
    self.printClasses()
    

  def setModelParameters(self,parameters):
    self.trainingSettings = parameters


  def initiateGenerators(self):
    a=self.trainingSettings["image_data_generator"]
   
    datagen=ImageDataGenerator(
        rotation_range=a["rotation_range"] if "rotation_range" in a else 0,
        shear_range=a["shear_range"] if "shear_range" in a else 0.0,
        zoom_range=a["zoom_range"] if "zoom_range" in a else 0.0,
        width_shift_range=a["width_shift_range"] if "width_shift_range" in a else 0.0,
        height_shift_range=a["height_shift_range"] if "height_shift_range" in a else 0.0,
        horizontal_flip=a["horizontal_flip"] if "horizontal_flip" in a else False,
        vertical_flip=a["vertical_flip"] if "vertical_flip" in a else False,
        preprocessing_function=self.preProcess,
        validation_split=self.trainingSettings["split"]["validation"]
    )
    
    self.validation_generator=datagen.flow_from_dataframe(
        classes=self.classList,
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol,
        class_mode="categorical",
        target_size=self.trainingSettings["model_target_size"],
        shuffle=True,
        batch_size=self.trainingSettings["batch_size"],
        interpolation="nearest",
        subset="validation"
    )

    test_datagen=ImageDataGenerator(
        rescale=1./255.
    )

    self.test_generator=test_datagen.flow_from_dataframe(
        classes=self.classList,
        dataframe=self.testdf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol,
        shuffle=False,
        class_mode="categorical",
        target_size=self.trainingSettings["model_target_size"],
#        shuffle=False,
        batch_size=self.trainingSettings["batch_size"],
#        interpolation="nearest"
    )

  def evaluateModel(self):
    print("evaluation (test_generator)")
    self.test_generator.reset()
    scores = self.model.evaluate_generator(
        self.test_generator,
        self.test_generator.n // self.test_generator.batch_size+1,
        verbose=1)
        
    print(self.model.metrics_names[0], scores[0],self.model.metrics_names[1], scores[1])
    self.logger.info("evaluation score (test): {}".format(set(zip(self.model.metrics_names,scores))))    


    print("evaluation (validation_generator)")
    self.validation_generator.reset()
    scores = self.model.evaluate_generator(
        self.validation_generator,
        self.validation_generator.n // self.validation_generator.batch_size+1,
        verbose=1)
        
    print(self.model.metrics_names[0], scores[0],self.model.metrics_names[1], scores[1])
    self.logger.info("evaluation score (validation): {}".format(set(zip(self.model.metrics_names,scores))))    
    




    

  def confusionMatrix(self,save=False):
    if not self.ran_predictions:
      self._runPredictions()
      
    print(self.labels_ground_truths)
    print(self.labels_predictions)
    print(len(self.labels_ground_truths))
    print(len(self.labels_predictions))
   
    print('confusion matrix')
    print(confusion_matrix(self.labels_ground_truths, self.labels_predictions))
   

  def classificationReport(self,save=False):
    if not self.ran_predictions:
      self._runPredictions()
    
    print('classification report')
    print(classification_report(self.labels_ground_truths, self.labels_predictions))





  def _runPredictions(self):
    print('running predictions')


#    self.validation_generator.reset()
      
#    self.labels_ground_truths=[]
    # a = iter(self.validation_generator.filenames)
#    for c in self.validation_generator.classes:
#      for label, index in self.validation_generator.class_indices.items():
#          if index==c:
#              # class_index, class, filename
#              # print(c, label, next(a)) 
#              # print(c, label)
#              self.labels_ground_truths.append(label)


    # predictions
    # y_pred = class index for all n predictions
#    Y_pred = self.model.predict_generator(
#        self.validation_generator,
#        self.validation_generator.n // self.validation_generator.batch_size+1,
#        verbose=1)


    self.validation_generator.reset()
    Y_pred = self.model.predict_generator(
        self.test_generator,
        self.test_generator.n // self.test_generator.batch_size+1,
        verbose=1)



    Y_pred = Y_pred[:self.test_generator.n]
    y_pred = np.argmax(Y_pred, axis=1)

#    self.labels_predictions=[]
#    for index in y_pred:
#      self.labels_predictions.append(self.resolveClassId(index))

    self.ran_predictions = True

    self.labels_ground_truths = self.test_generator.classes
    self.labels_predictions = y_pred







  def whatever(self):
    datagen=ImageDataGenerator(preprocessing_function=self.preProcess)
    
    self.validation_generator3=datagen.flow_from_dataframe(
        classes=self.classList,
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol,
        class_mode=None,
        target_size=self.trainingSettings["model_target_size"],
        shuffle=False,
        batch_size=self.trainingSettings["batch_size"],
        interpolation="nearest"
        )

    print("evaluation (predict_generator)")
    probabilities = self.model.predict_generator(self.validation_generator3, 100)
    print(probabilities)


  def setBatchPredictTestPath(self,path):    
    self.currentTestPath = path


  def setBatchPredictTrainingSetSample(self,fraction=0.002):
    self.testdf = self.traindf.sample(frac=fraction).reset_index()
    self.setBatchPredictTestPath(self.testImageFolder)
   

  def setBatchPredictTestImage(self,path,label=""):
    if self.testdf is None or len(self.testdf.index)==0:
      self.testdf = pd.DataFrame([[path,label.encode('utf-8')]],columns=["image","label"])
    else:
      self.testdf = self.testdf.append({"image": path, "label":  label.encode('utf-8') }, ignore_index=True)


  def batchPredict(self):
    print("batch sample")
    print(self.testdf.head())

    for x,y in self.testdf.iterrows():
      testImage = os.path.join(self.currentTestPath,y["image"])
      img = image.load_img(testImage, target_size=self.trainingSettings["model_target_size"])
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      preds = self.model.predict(x)
      index = np.argmax(preds[0])
      certainty = np.max(preds[0])

      this_class = self.resolveClassId(index)

      print("{} {}: {} / label: {} ({})".format("1" if this_class==y.label else "0", y["image"], this_class, y.label, certainty))
  

  def batchPredict2(self):
    filenames = self.test_generator.filenames
    nb_samples = len(filenames)

    print(filenames)
    
    predictions = self.model.predict_generator(self.test_generator,steps=nb_samples,verbose=1)
    print(predictions)
    pred_argmax = np.argmax(predictions,axis=1)
    print(pred_argmax.shape)
    print(pred_argmax)
    
    self.readClasses()
    print(self.classes)

    self.logger.info("pred_argmax: {}".format(pred_argmax))
    self.logger.info("pred_argmax.shape: {}".format(pred_argmax.shape))


    for idx, val in enumerate(np.argmax(predictions,axis=1)):
#        print(idx, filenames[idx], val, predictions[val])
        print(idx, filenames[idx], val, predictions[idx][val], self.classes.loc[self.classes["id"]==val]["label"])


    p = self.model.predict()
    print(p)





if __name__ == "__main__":
  
  settings_file=utilities.utilities.getSettingsFilePath()
  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()

  try:
    model_name = sys.argv[1]
    model_version = sys.argv[2]
  except IndexError:
    print("need a model name and version as parameters")
    print("run 'python model_repo.py' for available models in project {}".format(settings["project_name"]))
    exit(0)

  
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'testing',logging.INFO)
  params=model_parameters.modelParameters()

  params.setModelParameters(
      model_name=settings["models"]["basename"]+"_150min",
      minimum_images_per_class=150,
      batch_size=32,
      split =  { "validation": 0.2, "test" : 0.1 },
      early_stopping={ "use": False, "monitor": "val_acc", "patience": 3, "verbose": 0, "restore_best_weights": True},
      save={ "after_every_epoch": True, "after_every_epoch_monitor": "val_acc", "when_configured": True },
  )

  tester = modelTest(project_settings=settings,logger=logger)
  tester.setModelParameters(params.getModelParameters())

  tester.initTesting()

  tester.setModelName(model_name)
  tester.setModelVersionNumber(model_version)

  tester.setModelArchitecture("InceptionV3")
  tester.loadModel()
  tester.initiateGenerators()

  tester.evaluateModel()
  tester.classificationReport()
  tester.confusionMatrix()
##  tester.whatever()
  

  


#  # batch predict
#  if True:
#    tester.setBatchPredictTrainingSetSample(0.015)
#  else:
#    tester.setBatchPredictTestPath("/storage/data/corvidae/images/")
#    tester.setBatchPredictTestImage(label="Pica pica pica (Linnaeus, 1758)",path="RMNH.AVES.56402_1.jpg")
#    tester.setBatchPredictTestImage(label="Garrulus glandarius glandarius (Linnaeus, 1758)",path="RMNH.AVES.140137_2.jpg")
#    tester.setBatchPredictTestImage(label="Corvus frugilegus frugilegus Linnaeus, 1758",path="RMNH.AVES.67832_1.jpg")
#  
#  tester.batchPredict()
  

#  tester.setImage()
#  tester.lastStepHeatmap()
#  tester.halfwayFeatureMaps()
#  tester.afgeleideDoorHetHeleModel()


