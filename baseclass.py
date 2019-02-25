#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:12:29 2019

@author: maarten
"""
import os
import pandas as pd
from keras.models import load_model
import datetime

class baseClass:
  
  project_settings = {}
  projectRoot = ""
  modelRepoFolder = ""
  modelName = ""
  model = ""
  modelFilePath = ""
  modelArchitecture = ""
  modelTargetSize = ()
  classListPath = ""
  classListPathEncoding = "utf-8-sig"
  testImageFolder = ""
  imageListFile = ""
  imageListFileEncoding = ""
  imageListClassCol = "label"
  imageListImageCol = "image"
  traindf = {}
  modelVersionNumber = "1"


  
  def setSettings(self,project_settings):
    self.project_settings = project_settings
    self._setProjectName()
    self._setProjectRoot()
    self._setImageFolder()
    self._setModelRepoFolder()
    self._setClassListPath()
    self._setImageList()
    self._setTestImageFolder()


  def getSetting(self,setting):
    if setting in self.project_settings:
      return self.project_settings[setting]
    else:
      raise ValueError("missing setting".format(setting))


  def getDownloadedListFile(self):
    return os.path.join(self.projectRoot, self.getSetting('output_lists')['downloaded_list'])


  def _setProjectName(self):
    self.projectName = self.getSetting('project_name')


  def _setProjectRoot(self):
    self.projectRoot = self.getSetting('project_root')


  def _setImageFolder(self):
    self.imageFolder = os.path.join(self.projectRoot, self.getSetting('image_download')['folder'])


  def _setModelRepoFolder(self):
    self.modelRepoFolder = os.path.join(self.projectRoot, self.getSetting('models')['folder'])
 
  
  def _setClassListPath(self):
    self.classListPath = os.path.join(self.projectRoot, self.getSetting('output_lists')['class_list'])


  def _setImageList(self):
    self.imageListFile = os.path.join(self.projectRoot,self.getSetting('output_lists')['downloaded_list'])
    self.imageListFileEncoding = self.getSetting('output_lists')['image_list']['encoding']
    self.imageListClassCol = self.getSetting('output_lists')['image_list']['col_class']
    self.imageListImageCol = self.getSetting('output_lists')['image_list']['col_image']



  def getProjectModels(self):
    if not self.modelRepoFolder == "":
      self.availableModels = [f for f in os.listdir(self.modelRepoFolder) if os.path.isfile(os.path.join(self.modelRepoFolder, f))]      


  def listProjectModels(self):
    self.getProjectModels()
    print("models in {}:".format(self.modelRepoFolder))
    for model in self.availableModels:
      mod_timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(self.modelRepoFolder,model)))
      statinfo = os.stat(os.path.join(self.modelRepoFolder,model))
      print("- {} ({}; {}mb)".format(model,mod_timestamp,round(statinfo.st_size/1e6)))
      

  def setModelVersionNumber(self,number):
    self.modelVersionNumber = number
    self._setModelFilePath()


  def setModelName(self,modelName):
    self.modelName = modelName 
    self.modelName.replace(".hd5","")
    self._setModelFilePath()

  
  def _setModelFilePath(self):
    self.setModelFilePath(os.path.join(self.modelRepoFolder,"{}_{}.{}".format(self.modelName,self.modelVersionNumber,'hd5')))


  def setModelFilePath(self,path):
    self.modelFilePath = path


  def setImageDownloadFolder(self,settings):
    if 'image_download' in settings and 'folder' in settings['image_download']:
      self.imgDownloadFolder = os.path.join(self.projectRoot, self.settings['image_download']['folder'])

  
  def setModelArchitecture(self,arch):
    self.modelArchitecture = arch


  def setModelTargetSize(self,model_target_size):
    self.modelTargetSize = model_target_size


  def _setTestImageFolder(self):
    if self.getSetting('test_image_folder'):
      self.testImageFolder = os.path.join(self.projectRoot,self.getSetting('test_image_folder'))
    else:
      raise ValueError("test image folder (test_image_folder) missing from settings")

      
  def getBaseSettings(self):
      return { "project_root": self.projectRoot,
               "model_name": self.modelName,
               "model_architecture": self.modelArchitecture,
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


  def loadModel(self,showSummary=False):
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
        

  def readImageListFile(self):
    sep=','
    f = open(self.imageListFile, "r")
    line = f.readline()
    if line.count('\t')>0:
      sep='\t'
    self.traindf=pd.read_csv(self.imageListFile,encoding=self.imageListFileEncoding,sep=sep)
   
#    grouped_df=self.traindf.groupby(by=['label'])
#    print(grouped_df)
#    print(len(grouped_df))
#    for key, item in grouped_df:
#      print([key,len(grouped_df.get_group(key))])
##      print(grouped_df.get_group(key), "\n\n")


    
