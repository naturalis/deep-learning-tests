#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:36:18 2019

@author: maarten
"""
import os
from keras.models import load_model
from keras.preprocessing import image
import logging
import numpy as np
import pandas as pd
import helpers.logger
import helpers.settings_reader
import baseclass


class predict(baseclass.baseClass):
  
  testImageRootFolder = ""
  testImages = []

  def __init__(self,settings,logger=None):
    if logger is not None:
      self.logger = logger
    self.settings = settings
    self.setProjectRoot(settings)
    self.setModelRepoFolder(settings)
    self.setClassListPath(settings)
  

  def loadModel(self,showSummary=True):
    if self.modelFilePath=="" or not os.path.isfile(self.modelFilePath):
      raise ValueError("no model or model file doesn't exist ({})".format(self.modelFilePath))
    else:
      self.model = load_model(self.modelFilePath)

    if showSummary:
      self.model.summary()    


  def setTestImageRootFolder(self,path):
    self.testImageRootFolder = path


  def setTestImage(self,file):
    self.testImages.append(file)


  def _readClasses(self):
    self.classes=pd.read_csv(self.classListPath, encoding=self.classListPathEncoding)
    
    
  def _resolveClassId(self,id):
    return self.classes.loc[self.classes['id'] == id].label


  def doPredictions(self):
    self._readClasses()
    
    for item in self.testImages:
      testImage = os.path.join(self.testImageRootFolder,item)
      if os.path.isfile(testImage):
        self.logger.info("test image {}".format(testImage))
        self._predict(testImage)
      else:
        self.logger.error("test image {} doesn't exist".format(testImage))

 
  def _predict(self,testImage):
    if self.modelArchitecture=="Inception":
      from keras.applications.inception_v3 import preprocess_input
    elif self.modelArchitecture=="Xception":
      from keras.applications.xception import preprocess_input
    else:
      raise ValueError("didn't load preprocess_input for architecture {}".format(self.modelArchitecture))
    
    # https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras
    img = image.load_img(testImage, target_size=self.modelTargetSize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
 
    
    preds = self.model.predict(x,verbose=1)

    index = np.argmax(preds[0])
    bla = np.max(preds[0])
#    print(index)
    print(testImage)
    print(bla)
    print(self._resolveClassId(index))
    print("")



if __name__ == "__main__":
#  settings_file = "./config/martin-collectie.yml"
  settings_file = "./config/aliens.yml"
  settings = helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger('./log/','predict',logging.INFO)

  predict = predict(settings, logger)
  predict.setModelName("alien_predator_1")
  predict.loadModel()
  
  predict.setModelArchitecture("Inception") # Inception, Xception
  predict.setModelTargetSize((299, 299)) # must match model's
  
  test_image_folder = "/storage/data/alien-predator/test/"

  predict.setTestImageRootFolder(test_image_folder) # optional
  
  if not test_image_folder=="":
    for filename in os.listdir(test_image_folder):
      predict.setTestImage(filename) # may be full path, will be joined with testImageRootFolder

  predict.doPredictions()

