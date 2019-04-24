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
  
  projectName = None
  project_settings = {}
  projectRoot = ""
  modelRepoFolder = ""
  modelName = ""
  model = ""
  modelFilePath = ""
  modelJsonFilePath = ""
  modelArchitecture = ""
  modelTargetSize = ()
  classListPath = ""
  classListPathEncoding = "utf-8-sig"
  testImageFolder = ""
  imageListFile = ""
  imageListFileEncoding = ""
  imageListClassCol = "label"
  imageListImageCol = "image"
  traindf = None
  testdf = None
  modelVersionNumber = "1"
  classList = []
  availableModels = []
  trainingSettings = {}
    

  def setSettings(self,project_settings):
    self.project_settings = project_settings
    self._setProjectName()
    self._setProjectRoot()
    self._setImageFolder()
    self._setModelRepoFolder()
    self._setTestImageFolder()
    self._setClassListPath()


  def getSetting(self,setting):
    if setting in self.project_settings:
      return self.project_settings[setting]
    else:
      pass
#      raise ValueError("missing setting".format(setting))


  def getImageList(self,list_type):
    if not list_type in self.getSetting('image_lists'):
      raise ValueError("unknown list type: {}".format(list_type))
    a = self.getSetting('image_lists')[list_type]
    a['path'] = os.path.join(self.projectRoot,a['file'])
    return a


  def setModelVersionNumber(self,number):
    self.modelVersionNumber = number
    self._setModelFilePath()


  def setModelName(self,modelName):
    self.modelName = modelName 
    self.modelName.replace(".hd5","")
    self._setModelFilePath()


  def readTrainAndTestImageLists(self):
    try:
      test_split = self.trainingSettings["split"]["test"]
    except:
      test_split = -1

    if test_split > 0:
      self.traindf = self.readImageListFile(self.getImageList("train")["path"])
      self.testdf = self.readImageListFile(self.getImageList("test")["path"])
      self.logger.info("read {}; got {} training images".format(self.getImageList("train")["path"],len(self.traindf)))
      self.logger.info("read {}; got {} test images".format(self.getImageList("test")["path"],len(self.testdf)))
      self.useTestSplit = True
    else:
      self.readDownloadedImagesFile()
      self.testdf = None
      self.logger.info("read infile {}; got {} training images".format(self.imageListFile,len(self.traindf)))
      self.logger.info("no test images")
      self.useTestSplit = False
     
#    print(self.testdf.head())
#    print(self.traindf.head())
     


  def _setProjectName(self):
    self.projectName = self.getSetting('project_name')


  def _setProjectRoot(self):
    self.projectRoot = self.getSetting('project_root')


  def _setImageFolder(self):
    self.imageFolder = os.path.join(self.projectRoot, self.getSetting('image_download')['folder'])


  def _setModelRepoFolder(self):
    self.modelRepoFolder = os.path.join(self.projectRoot, self.getSetting('models')['folder'])
 

  def _setClassListPath(self):
    self.classListPath = self.getImageList("classes")["path"]


  
  def _setModelFilePath(self):
    self.setModelFilePath(os.path.join(self.modelRepoFolder,"{}_{}.{}".format(self.modelName,self.modelVersionNumber,'hd5')))
    self.setModelJsonFilePath(os.path.join(self.modelRepoFolder,"{}_{}.{}".format(self.modelName,self.modelVersionNumber,'json')))


  def setModelFilePath(self,path):
    self.modelFilePath = path


  def setModelJsonFilePath(self,path):
    self.modelJsonFilePath = path


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
    self.classes = pd.read_csv(self.classListPath, encoding=self.classListPathEncoding)
    self.classList = list(self.classes["label"])


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
        
  
  def readImageListFile(self,file_path,encoding="utf-8-sig"):
    f = open(file_path, "r", encoding=encoding)
    line = f.readline()
    sep=','
    if line.count('\t')>0:
      sep='\t'
    return pd.read_csv(file_path, encoding=encoding, sep=sep, dtype="str")


  def readDownloadedImagesFile(self):
    list_file = self.getImageList("downloaded")
    self.traindf = self.readImageListFile(list_file["path"])


  def saveClassList(self,dataFrame):
    classes = []

    try:
      a = dataFrame.groupby(by=1)
    except KeyError:
      a = dataFrame.groupby(by="label")

    for key, item in a:      
      classes.append(([key, len(a.get_group(key))]))
       
    a = pd.DataFrame(classes, columns = ["label", "image_count"])
    a = a.sort_values(by=["label"]).reset_index(drop=True)

    list_file = self.getImageList("classes")
    a.to_csv(list_file["path"], index=False)

    self.logger.info("wrote {} classes to {}".format(len(a),list_file["path"]))
