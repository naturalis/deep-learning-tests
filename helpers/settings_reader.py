#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 16:49:23 2019

@author: maarten
"""

import os.path
import yaml

class settingsReader:

  settingsFile = "settings.yml"
  settings = {}

  def __init__(self,settingsFile=None):
    if settingsFile is not None:
      self.settingsFile = settingsFile

    if not os.path.isfile(self.settingsFile):
      raise FileNotFoundError("settings file doesn't exist: {}".format(self.settingsFile))
      exit(1)
      
    self._readSettings()

  def getSettings(self):
    return self.settings    

  def _readSettings(self):
    stream = open(self.settingsFile, "r")
    docs = yaml.load_all(stream)
    for doc in docs:
      for k,v in doc.items():
        self.settings[k]=v
