#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten.schermer@naturalis.nl
"""
import baseclass
import utilities
import logging, os, glob
import helpers.logger
import helpers.settings_reader
import datetime

class modelRepo(baseclass.baseClass):
  
  testdf = None
  ran_predictions = False
  
  def __init__(self, project_settings, logger):
    self.logger = logger
    self.setSettings(project_settings)

  def getProjectModels(self):
    if not self.modelRepoFolder == "":
      os.chdir(self.modelRepoFolder)
      for file in glob.glob("*.hd5"):
        self.availableModels.append(file)


  def listProjectModels(self):
    self.getProjectModels()
    print("models in {}:".format(self.modelRepoFolder))
    for model in self.availableModels:
      mod_timestamp = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(self.modelRepoFolder,model))).strftime('%Y-%m-%d %X')
      statinfo = os.stat(os.path.join(self.modelRepoFolder,model))
      
      version = model.split("_").pop()
      model_name = model.replace("_"+version ,"")
      version = version.replace(".hd5","")
      
      print("* {} {} ({}; {}mb)".format(model_name,version,mod_timestamp,round(statinfo.st_size/1e6)))


if __name__ == "__main__":
  settings_file=utilities.utilities.getSettingsFilePath()

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'testing',logging.INFO)
  
  repo = modelRepo(project_settings=settings,logger=logger)
  repo.listProjectModels()
