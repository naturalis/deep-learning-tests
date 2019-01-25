#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 11:36:18 2019

@author: maarten
"""
import os
from keras.preprocessing import image
import logging
import numpy as np
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


  def setTestImageRootFolder(self,path):
    self.testImageRootFolder = path


  def setTestImage(self,file):
    self.testImages.append(file)

    
  def doPredictions(self):
    self.readClasses()
    
    for item in self.testImages:
      testImage = os.path.join(self.testImageRootFolder,item)
      if os.path.isfile(testImage):
        self.logger.debug("loaded test image {}".format(testImage))
        self._predict(testImage)
      else:
        self.logger.error("test image {} doesn't exist".format(testImage))

 
  def _predict(self,testImage):
    
#    preprocess_input(img, model = c("Xception", "VGG16", "VGG19", "ResNet50","InceptionV3"))
    
    
    
    if self.modelArchitecture=="InceptionV3":
      from keras.applications.inception_v3 import preprocess_input, decode_predictions
    elif self.modelArchitecture=="Xception":
      from keras.applications.xception import preprocess_input, decode_predictions
    elif self.modelArchitecture=="VGG16":
      from keras.applications.vgg16 import preprocess_input, decode_predictions
    else:
      raise ValueError("unknown architecture {}".format(self.modelArchitecture))
    
    # https://stackoverflow.com/questions/47555829/preprocess-input-method-in-keras
    img = image.load_img(testImage, target_size=self.modelTargetSize)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
 
    preds = self.model.predict(x)
    index = np.argmax(preds[0])
    certainty = np.max(preds[0])
    this_class = self.resolveClassId(index)

    print("{}: {} ({})".format(testImage.replace(self.testImageRootFolder,""), 
          this_class, 
          certainty))


if __name__ == "__main__":
#  settings_file = "./config/martin-collectie.yml"
  settings_file = "./config/aliens.yml"
  settings = helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger('./log/','predict',logging.DEBUG,False)

  predict = predict(settings, logger)
  predict.setModelName("alien_predator_inception_plus")
  predict.loadModel()

  predict.setModelArchitecture("InceptionV3") # Inception, Xception, VGG16
  predict.setModelTargetSize((299, 299)) # must match model's
  
  test_image_folder = "/storage/data/alien-predator/test/"

  predict.setTestImageRootFolder(test_image_folder) # optional
  
  if not test_image_folder=="":
    for filename in os.listdir(test_image_folder):
      predict.setTestImage(filename) # may be full path, will be joined with testImageRootFolder

  predict.doPredictions()
  predict.printClasses()


# confusion matrix
# list of all test images + their predictions
# image of intermediate activations