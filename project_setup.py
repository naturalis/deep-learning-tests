#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 15:42:37 2019

@author: maarten
"""

import baseclass
import os.path
import logging
import helpers.logger
import helpers.settings_reader

class projectSetup(baseclass.baseClass):

  def __init__(self,project_settings,logger=None):
    if logger is not None:
      self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))


  def runSetup(self):
    self._createProjectFolder()
    self._createImageFolder()
    self._createDataFolder()


  def _createProjectFolder(self):
    self._createFolder(self.projectRoot,"project")


  def _createImageFolder(self):
    self._createFolder(self.imageFolder,"download")


  def _createDataFolder(self):
    self._createFolder(os.path.join(self.projectRoot,"data/"),"data")


  def _createFolder(self,path,label):
    if not os.path.exists(path):
      os.makedirs(path)
      self.logger.info("created {} folder: {}".format(label,path))
    else:
      self.logger.info("{} folder exists: {}".format(label,path))



if __name__ == "__main__":
  settings_file = "./config/corvidae.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)

  setup = projectSetup(settings,logger)
  setup.runSetup()
