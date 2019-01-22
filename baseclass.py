#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 10:12:29 2019

@author: maarten
"""
import os

class baseClass:

  projectRoot = ""
  modelRepoFolder = ""
  modelName = ""
  modelFilePath = ""
  modelArchitecture = ""
  modelTargetSize = ()
  classListPath = ""
  classListPathEncoding = "utf-8-sig"

  def setProjectRoot(self,settings):
    if 'project_root' in settings:
      self.projectRoot = settings['project_root']
    else:
      raise ValueError("project_root missing from settings")


  def setModelRepoFolder(self,settings):
    if 'models' in settings and 'folder' in settings['models']:
      self.modelRepoFolder = os.path.join(self.projectRoot,self.settings['models']['folder'])
    else:
      raise ValueError("model repo (models:folder) missing from settings")
 
  
  def setModelName(self,modelName):
    self.modelName  = modelName 
    self._makeModelFilePath()

  
  def _makeModelFilePath(self):
    self.modelFilePath = os.path.join(self.modelRepoFolder,self.modelName + '.hd5')

  
  def setModelArchitecture(self,arch):
    self.modelArchitecture = arch


  def setModelTargetSize(self,dim):
    self.modelTargetSize = dim


  def setClassListPath(self,settings):
    if 'output_lists' in settings and 'class_list' in settings['output_lists']:
      self.classListPath = os.path.join(self.projectRoot,settings['output_lists']['class_list'])
    else:
      raise ValueError("classlist path (output_lists:class_list) missing from settings")
      
