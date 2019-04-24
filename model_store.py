#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten
"""

import baseclass
import utilities
import hashlib
import time
import os
import logging
import model_parameters
import helpers.logger
import helpers.settings_reader

class modelStore(baseclass.baseClass):

  projectname = None;
  versionnumber = None;
  imagelist = None;
  modeldefinition = None;
  modelparameters = None;
  trainingparameters = None;
  modeldefinition = None;
  imagelisttrain = None;
  imagelisttest = None;

      
  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))


  def getVersionNumber(self):
    m = hashlib.md5()
    m.update(str(round(time.time())).encode("utf-8-sig"))
    return m.hexdigest()
  
  def storeModelPackage(self):
    self._checkMandatory()

  def setProjectName(self,projectname):
    self.projectname = projectname

  def setImagelistTrain(self,imagelist):
    self.imagelisttrain = imagelist

  def setImagelistTest(self,imagelist):
    self.imagelisttest = imagelist

  def setModelDefinition(self,modeldefinition):
    # architectural definition
    self.modeldefinition = modeldefinition

  def setModelParameters(self,modelparameters):
    # all non-phase definitions
    self.modelparameters = modelparameters

  def setTrainingParameters(self,trainingparameters):
    # training phases
    self.trainingparameters = trainingparameters

  def _checkMandatory(self):
    if self.projectname is None:
        raise ValueError("no project name provided")
    if self.modelVersionNumber is None:
        raise ValueError("no version number provided")
    if self.imagelisttrain is None:
        raise ValueError("no training image list provided")
    if self.modeldefinition is None:
        raise ValueError("no model definition provided")
    if self.modelparameters is None:
        raise ValueError("no model parameters provided")
    if self.trainingparameters is None:
        raise ValueError("no training parameters provided")
    if self.modeldefinition is None:
        raise ValueError("no model definition provided")




#    model
#    params
#    results
#    datetime
#    dataset
#     files + original URL
#     split
#   short description
#   notes
#   results



if __name__ == "__main__":
  settings_file=utilities.utilities.getSettingsFilePath()
  settings = helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)












  exit(0)




  
  params = model_parameters.modelParameters()
  params.setModelParameters(
      model_name=settings["models"]["basename"],
      minimum_images_per_class=150,
      batch_size=64,
      split = { "validation": 0.2, "test" : 0.1 },
#      early_stopping={ "use": True, "monitor": "val_acc", "patience": 5, "verbose": 0, "restore_best_weights": True },
      save={ "after_every_epoch": True, "after_every_epoch_monitor": "val_acc", "when_configured": True },
  )

  version = '1350b3beb093a1f6a69d4ea036fdebed'
  
  # SHOULD BE parameters["model_architecture]"
  model_architecture = 'InceptionV3'



  store = modelStore(project_settings=settings, logger=logger)

  store.setSettings(settings)  
  store.setModelParameters(params.getModelParameters())

  store.setProjectName(settings["project_name"]) 
  store.setModelArchitecture(model_architecture)
  store.setModelVersionNumber(version)

  

  
  
  store.readTrainAndTestImageLists()
  store.setImagelistTrain(store.traindf)
  store.setImagelistTest(store.testdf)
  
  print(store.modelJsonFilePath)
  
  store.setModelDefinition()

  store.storeModelPackage()

  

# store.setImagelist(imagelist)
# store.setImagelist(imagelist)
# store.setImagelist(imagelist)
# store.setImagelist(imagelist)
# store.setImagelist(imagelist)
# store.setImagelist(imagelist)
# store.setImagelist(imagelist)
# store.setImagelist(imagelist)
# store.setImagelist(imagelist)
