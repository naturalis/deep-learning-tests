#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 16:42:01 2018

@author: maarten
"""
import os, sys, csv
import logging
import logger
from os import walk
from PIL import Image

class imageCheck:
  
  folders = []
  images = []
  brokenFiles = []
  readRecursively = False
  extensions = ( ".jpg" )

  def __init__(self,logger=None):
    if logger is not None:
      self.logger = logger
  
  def addFolder(self,folder):
    self.folders.append(folder)

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

  def writeCorruptFiles(self,outfile,includeError=False):
      with open(outfile, 'w+') as file:
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
  logger = logger.logger('dwca_reader',logging.INFO)
  d = imageCheck(logger)
  d.setExtensions((".jpg",".png"))
  d.addFolder("/storage/deep-learning-experiments/martin-collectie/downloads/")
  d.setReadFoldersRecursive(True)
  d.checkFiles()
  d.writeCorruptFiles("./corrupt_files.csv",False)
  
