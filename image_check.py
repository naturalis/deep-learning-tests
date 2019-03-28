#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:42:01 2018

@author: maarten
"""
import baseclass
import os, sys, csv
import logging
import helpers.logger
import helpers.settings_reader
from os import walk
from PIL import Image

class imageCheck(baseclass.baseClass):
  
  folders = []
  images = []
  brokenFiles = []
  readRecursively = False
  extensions = ( ".jpg" )

  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))
    self.addFolder(self.imageFolder)
    
    try:
      list_file = self.getImageList("corrupt")
      self.corruptImagesFile = list_file["path"]
    except:
      self.corruptImagesFile = os.path.join(self.projectRoot,"corrupt_images.csv")


  def addFolder(self,folder):
    self.folders.append(folder)
    self.logger.info("added folder {}".format(folder))


  def setExtensions(self,extensions):
    self.extensions = extensions


  def setReadFoldersRecursive(self,state):
    self.readRecursively = state


  def checkFiles(self):
    self.brokenFiles=[]
    self._readFiles()
    ok = 0
    nok = 0
    for image in self.images:
      try:
        im = Image.open(image)
        im.verify()
        ok += 1
      except:
        print("{}".format(sys.exc_info()[1]))
        self.brokenFiles.append((image,str(sys.exc_info()[1])))
        nok += 1
    
    print("verified {}: {} ok, {} not ok".format(len(self.images),ok,nok))


  def writeCorruptFiles(self,includeError=False):
      with open(self.corruptImagesFile, 'w+') as file:
        c = csv.writer(file)
        for item in self.brokenFiles:
          if includeError:
            c.writerow([item[0],item[1]])
          else:
            c.writerow([item[0]])


  def _readFiles(self):
    for folder in self.folders:
      for (dirpath, dirnames, files) in walk(folder):
        for file in files:
            if file.endswith(self.extensions):
                self.images.append(os.path.join(dirpath, file))
        if not self.readRecursively:
          # if you only want the top directory, break the first time it yields
          break    


if __name__ == "__main__":
  settings_file = "./config/corvidae.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'dwca_reader',logging.INFO)

  d = imageCheck(settings, logger)
  d.setExtensions((".jpg",".png"))
  # d.addFolder("/path/to/extra/folder/")
  d.setReadFoldersRecursive(True)
  d.checkFiles()
  d.writeCorruptFiles()
  


