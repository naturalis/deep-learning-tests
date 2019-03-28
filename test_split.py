#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten.schermer@naturalis.nl
"""
import baseclass
import model_parameters
import os, csv
import logging
import helpers.logger
import helpers.settings_reader
from sklearn.model_selection import StratifiedShuffleSplit

class testSplit(baseclass.baseClass):

  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))


  def setModelParameters(self,parameters):    
    self.trainingSettings["split"]=parameters["split"]


  def doTestSplit(self):
    self.readDownloadedImagesFile()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=self.trainingSettings["split"]["test"], random_state=0)

    train_ids = []
    test_ids = []

    for train_index, test_index in sss.split(self.traindf["image"], self.traindf["label"]):
      train_ids.extend(train_index)
      test_ids.extend(test_index)
    
    testdf = self.traindf.ix[test_ids].reset_index()
    traindf = self.traindf.ix[train_ids].reset_index()
    
    grouped_traindf = traindf.groupby(by=['label'])
    grouped_testdf = testdf.groupby(by=['label'])

    list_file_train = self.getImageList("train")
    with open(list_file_train["path"], 'w') as file:
      c = csv.writer(file)
      c.writerow(["label","image"])
      for key, item in grouped_traindf:
        try:
          data = key.decode()
        except AttributeError:
          data = key
        
        for item in grouped_traindf.get_group(key)["image"].iteritems():
          c.writerow([data.encode("utf-8"),item[1]])

    list_file_test = self.getImageList("test")
    with open(list_file_test["path"], 'w') as file:
      c = csv.writer(file)
      c.writerow(["label","image"])
      for key, item in grouped_testdf:
        try:
          data = key.decode()
        except AttributeError:
          data = key

        for item in grouped_testdf.get_group(key)["image"].iteritems():
          c.writerow([data.encode("utf-8"),item[1]])

    self.logger.info("did {} test split: {} train, {} test".format(self.trainingSettings["split"]["test"],len(train_ids),len(test_ids)))
    self.logger.info("training images: {}".format(list_file_train["path"]))
    self.logger.info("test images: {}".format(list_file_test["path"]))



if __name__ == "__main__":
  settings_file="./config/corvidae.yml"
#  settings_file="./config/mnist.yml"

  settings = helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)
  params = model_parameters.modelParameters()

  params.setModelParameters( split = { "validation": 0.2, "test" : 0.1  } )
  
  split=testSplit(project_settings=settings, logger=logger)
  split.setModelParameters(params.getModelParameters())
  split.doTestSplit()
  
