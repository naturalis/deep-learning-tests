#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten
"""
import baseclass
import model_parameters
import logging, os
import numpy as np
import pandas as pd
import helpers.logger
import helpers.settings_reader
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import decode_predictions, preprocess_input
from sklearn.metrics import classification_report, confusion_matrix

# https://datascience.stackexchange.com/questions/32814/error-while-using-flow-from-generator
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

class modelTest(baseclass.baseClass):
  
  testdf = None
  
  def __init__(self, project_settings, logger):
    self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))
    self.readImageListFile()
    self.readClasses()


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
        validation_split=self.trainingSettings["split"]["validation"])
    
    self.validation_generator=datagen.flow_from_dataframe(
        dataframe=self.traindf, 
        directory=self.imageFolder,
        x_col=self.imageListImageCol, 
        y_col=self.imageListClassCol,
        class_mode="categorical",
        target_size=self.trainingSettings["model_target_size"],
        batch_size=self.trainingSettings["batch_size"],
        interpolation="nearest",
        subset="validation")

#    test_datagen = ImageDataGenerator(rescale=1./255)
#
#    self.test_generator=test_datagen.flow_from_dataframe(
#        dataframe=self.testdf,
#        directory=self.imageFolder,
#        x_col=self.imageListImageCol, 
#        y_col=None,
#        batch_size=1,
#        class_mode=None,
#        shuffle = False,
#        target_size=self.modelTargetSize
#        )


  def evaluateModel(self):
#    # in case of fit_generator, the evaluation is part of the fitting itself
#    scores = self.model.evaluate_generator(
#        self.test_generator,
#        self.test_generator.n/self.test_generator.batch_size,
#        verbose=1)
#
#    print("test_generator")
#    print(self.model.metrics_names[0], scores[0],self.model.metrics_names[1], scores[1])

    print("evaluation (validation_generator)")
    scores = self.model.evaluate_generator(
        self.validation_generator,
        self.validation_generator.n/self.validation_generator.batch_size,
        verbose=1)
    
    print(self.model.metrics_names[0], score2s[0],self.model.metrics_names[1], scores[1])
    self.logger.info("evaluation score: {}".format(set(zip(self.model.metrics_names,scores))))    


  def confusionMatrix(self):
    Y_pred = self.model.predict_generator(self.validation_generator, self.validation_generator.n // self.validation_generator.batch_size+1)
    y_pred = np.argmax(Y_pred, axis=1)

    print('confusion matrix')
    print(confusion_matrix(self.validation_generator.classes, y_pred))

    print('classification report')
    print(classification_report(self.validation_generator.classes, y_pred))


  def setBatchPredictTestPath(self,path):    
    self.currentTestPath = path


  def setBatchPredictTrainingSetSample(self,fraction=0.002):
    self.testdf = self.traindf.sample(frac=fraction).reset_index()
    self.setCurrentTestPath(self.testImageFolder)
   

  def setBatchPredictTestImage(self,path,label=""):
    if self.testdf is None or len(self.testdf.index)==0:
      self.testdf = pd.DataFrame([[path,label.encode('utf-8')]],columns=["image","label"])
    else:
      self.testdf = self.testdf.append({"image": path, "label": label.encode('utf-8') }, ignore_index=True)


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

      print("{}: {} ({}) / label: {}".format(y["image"], this_class, certainty, y.label))
  

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
#  settings_file="./config/martin-collectie.yml"
#  settings_file="./config/aliens.yml"
#  settings_file="./config/mnist.yml"
  settings_file="./config/corvidae.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)
  params=model_parameters.modelParameters()

  tester = modelTest(project_settings=settings,logger=logger)
  tester.setModelParameters(params.getModelParameters())
  tester.listProjectModels()

  tester.setModelName('corvidae_InceptionV3')
  tester.loadModel()
  tester.initiateGenerators()

  # confusion matrix & classification report, evaluations
  tester.confusionMatrix()
  tester.evaluateModel()

  
  # batch predict
  if False:
    tester.setBatchPredictTrainingSetSample(0.01)
  else:
    tester.setBatchPredictTestPath("/storage/data/corvidae/images/")
    tester.setBatchPredictTestImage(label="Pica pica pica (Linnaeus, 1758)",path="RMNH.AVES.56402_1.jpg")
    tester.setBatchPredictTestImage(label="Garrulus glandarius glandarius (Linnaeus, 1758)",path="RMNH.AVES.140137_2.jpg")
    tester.setBatchPredictTestImage(label="Corvus frugilegus frugilegus Linnaeus, 1758",path="RMNH.AVES.67832_1.jpg")
  
  tester.batchPredict()



  
#  tester.generalModelStats()
#  tester.performancePerClass()
#   precision
#   recall

#  tester.setImage()
#  tester.lastStepHeatmap()
#  tester.halfwayFeatureMaps()
#  tester.afgeleideDoorHetHeleModel()


