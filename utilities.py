#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:55:20 2019

@author: maarten
"""

import baseclass

class utilities(baseclass.baseClass):

  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    
  def doSaveClassList(self):
    a = self.readImageListFile(self.getImageList('downloaded')["path"])
    self.saveClassList(a)
