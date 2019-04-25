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
from datetime import datetime
import os
import hashlib
import json
import logging
import model_parameters
import helpers.logger
import helpers.settings_reader

class modelStore(baseclass.baseClass):

  projectname = None
  versionnumber = None
  modeldefinition = None
  modelparameters = None
  trainingparameters = None
  imagelisttrain = None
  imagelisttest = None
  evaluationscores = []
  notes = []
  timestore = None
  timemodel = None

  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))

  def generateVersionNumber(self):
    m = hashlib.md5()
    m.update(str(round(time.time())).encode("utf-8-sig"))
    return m.hexdigest()
  
  def storeModelPackage(self):
    self._checkMandatory()
    self._setTimeStamps()
    self._makePackage()
    self._savePackage()
    

  def setProjectName(self,projectname):
    self.projectname = projectname

  def setImagelistTrain(self,imagelist):
    self.imagelisttrain = imagelist

  def setImagelistTest(self,imagelist):
    self.imagelisttest = imagelist

  def setModelDefinition(self,modeldefinition):
    # architectural definition
    with open(modeldefinition, 'r') as myfile:
      data=myfile.read()

    self.modeldefinition = json.loads(data)

  def setModelParameters(self,modelparameters):
    # all non phase-dependent definitions
    self.modelparameters = modelparameters
    if "split" in self.modelparameters:
        store.trainingSettings["split"] = self.modelparameters["split"]

  def setTrainingParameters(self,trainingparameters):
    # training phases
    self.trainingparameters = trainingparameters

  def setEvaluationScore(self,score_validation):
    self.evaluationscores.append(score_validation)
    
  def setNote(self,label,note):
    self.notes.append({'label':label,'note':note})

  def _checkMandatory(self):
    if self.projectname is None:
        raise ValueError("no project name provided")

    if self.modelVersionNumber is None:
        raise ValueError("no version number provided")

#    if self.imagelisttrain is None:
#        raise ValueError("no training image list provided")

    if self.modeldefinition is None:
        raise ValueError("no model definition provided")

    if self.modelparameters is None:
        raise ValueError("no model parameters provided")

    if self.trainingparameters is None:
        raise ValueError("no training parameters provided")

    if self.modeldefinition is None:
        raise ValueError("no model definition provided")

  def _setTimeStamps(self):
    self.timestore = datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    self.timemodel = datetime.fromtimestamp(os.path.getmtime(store.modelJsonFilePath)).strftime('%Y-%m-%d %H:%M:%S')

  def _makePackage(self):
    self._package = {
      "projectname" : self.projectname,
      "model" : {
        "version" : self.modelVersionNumber,
        "definition" : self.modeldefinition,
        "parameters" : self.modelparameters      
      },
      "training" : self.trainingparameters,
      "images" : {      
        "train" : self.imagelisttrain,
        "test" : self.imagelisttest,
      }
    }

    self._package["package_hash"] = hashlib.sha1(json.dumps(self._package, sort_keys=True).encode('utf-8')).hexdigest()
    
    self._package["evaluation"] = {
        "scores" : self.evaluationscores
    }
    self._package["timestamps"] = {
        "store" : self.timestore,
        "model" : self.timemodel
    }
    self._package["notes"] = self.notes
    
  def _savePackage(self):
    print(json.dumps(self._package, sort_keys=True))


if __name__ == "__main__":

  settings_file=utilities.utilities.getSettingsFilePath()
  custom_params_file=utilities.utilities.getCustomParametersFilePath()

  settings = helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)
  params = model_parameters.modelParameters(logger=logger)
  params.readCustomSettingsFile(custom_params_file)

  # stubs
  version = '30bface30dbf28ab99a9e7b2181ccad4'
  score_validation = { 'label' : 'validation', 'scores' : { 'loss': 0.07475143565367257, 'acc': 0.9853333333333333}}
  score_test = { 'label' : 'test', 'scores' : { 'loss': 0.07734002753771779, 'acc': 0.9864} }
  note_label = "first thoughts"
  note = "it's a model and it's looking good"

  store = modelStore(project_settings=settings, logger=logger)
  store.setModelName(settings["models"]["basename"])
  store.setSettings(settings)  
  store.setModelParameters(params.getModelParameters())
  store.setProjectName(settings["project_name"]) 
  store.setModelArchitecture(params.parameters["model_architecture"])
  store.setModelVersionNumber(version)
  store.setModelDefinition(store.modelJsonFilePath)
  store.setTrainingParameters(params.parameters)  
  store.readTrainAndTestImageLists()
#  store.setImagelistTrain(store.traindf)
#  store.setImagelistTest(store.testdf)
  store.setEvaluationScore(score_validation)
  store.setEvaluationScore(score_test)
  store.setNote(note_label,note)

  store.storeModelPackage()
