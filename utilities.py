#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:55:20 2019

@author: maarten
"""

import baseclass
import os

class utilities(baseclass.baseClass):

  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    
  def doSaveClassList(self):
    a = self.readImageListFile(self.getImageList('downloaded')["path"])
    self.saveClassList(a)

  @staticmethod
  def getProjectNameEnv():
    if not "project" in os.environ:
      raise ValueError("no project name (use: export project=<project_name>)") 

    return os.environ["project"]

  @staticmethod
  def getSettingsFilePath(env_project=None):
    if env_project==None:
      env_project=utilities.getProjectNameEnv()
    return "./config/"+env_project+".yml"
  
  @staticmethod
  def getCustomParametersFilePath():
    env_project=utilities.getProjectNameEnv()
    return "./config/"+env_project+"_params.json"
  