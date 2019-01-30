#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:12:29 2019

@author: maarten
"""
import os
import pandas as pd
from keras.models import load_model

class baseClass:

  projectRoot = ""
  modelRepoFolder = ""
  modelName = ""
  modelFilePath = ""
  modelArchitecture = ""
  modelTargetSize = ()
  classListPath = ""
  classListPathEncoding = "utf-8-sig"
  testImageFolder = ""

  def setProjectRoot(self,settings):
    if 'project_root' in settings:
      self.projectRoot = settings['project_root']
    else:
      raise ValueError("project_root missing from settings")


  def setModelRepoFolder(self,settings):
    if 'models' in settings and 'folder' in settings['models']:
      self.modelRepoFolder = os.path.join(self.projectRoot,self.settings['models']['folder'])
    else:
      raise ValueError("model repo (models:folder) missing from settings")
 
  
  def setModelName(self,modelName):
    self.modelName  = modelName 
    self._makeModelFilePath()

  
  def _makeModelFilePath(self):
    self.modelFilePath = os.path.join(self.modelRepoFolder,self.modelName + '.hd5')

  
  def setModelArchitecture(self,arch):
    self.modelArchitecture = arch


  def setModelTargetSize(self,model_target_size):
    self.modelTargetSize = model_target_size


  def setClassListPath(self,settings):
    if 'output_lists' in settings and 'class_list' in settings['output_lists']:
      self.classListPath = os.path.join(self.projectRoot,settings['output_lists']['class_list'])
    else:
      raise ValueError("classlist path (output_lists:class_list) missing from settings")


  def setTestImageFolder(self,settings):       
    if 'test_image_folder' in settings:
      self.testImageFolder = os.path.join(self.projectRoot,settings['test_image_folder'])
    else:
      raise ValueError("test image folder (test_image_folder) missing from settings")

      
  def getBaseSettings(self):
      return { "project_root": self.projectRoot,
               "model_name": self.modelName,
               "model_architecture": self.modelArchitecture,
               "input_size": self.modelInputSize,
               "target_size": self.modelTargetSize }


  def getBaseFileSettings(self):
      return { "project_root": self.modelFilePath,
               "class_list_path": self.classListPath }


  def readClasses(self):
    self.classes=pd.read_csv(self.classListPath, encoding=self.classListPathEncoding)


  def resolveClassId(self,id):
    return self.classes.loc[self.classes['id'] == id].label.item()
  

  def printClasses(self):
    print(self.classes)


  def loadModel(self,showSummary=True):
    if self.modelFilePath=="" or not os.path.isfile(self.modelFilePath):
      raise ValueError("no model or model file doesn't exist ({})".format(self.modelFilePath))
    else:
      self.model = load_model(self.modelFilePath)
      
    self.logger.debug("{} {} {}".format(self.modelName,self.modelArchitecture,self.modelTargetSize))

    if showSummary:
      self.model.summary()    


  def preProcess(self,x):
      if self.modelArchitecture == "InceptionV3":
        import keras.applications.inception_v3
        return keras.applications.inception_v3.preprocess_input(x)
      elif self.modelArchitecture == "Xception":
        import keras.applications.xception
        return keras.applications.xception.preprocess_input(x)
      elif self.modelArchitecture == "VGG16":
        import keras.applications.vgg16
        return keras.applications.vgg16.preprocess_input(x)
      else:
        raise ValueError("unknown model architecture {}".format(self.modelArchitecture))
        

