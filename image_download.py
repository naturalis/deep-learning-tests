#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:54:07 2019

@author: maarten.schermer@naturalis.nl
"""

import baseclass
import csv
import os.path
import sys, re
import urllib.request
from urllib.parse import urlparse
import logging
import helpers.logger
import helpers.settings_reader
import numpy as np
import pandas as pd

class imageDownloader(baseclass.baseClass):
    
  headerCols = []
  imageDownloadList = []

  def __init__(self,project_settings,logger):
    self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))



  def downloadImages(self,skip_download_if_exists=False):    
    # image_url_path_parse
    if not self.getSetting('image_url_path_parse') == None:
      if 'reg_exp' in self.getSetting('image_url_path_parse'):
        self.imgUrlPathParseRegExp = self.getSetting('image_url_path_parse')['reg_exp']
  
      if 'name_index' in self.getSetting('image_url_path_parse'):
        self.imgUrlPathParseNameIdx = self.getSetting('image_url_path_parse')['name_index']

    if 'default_extension' in self.getSetting('image_download'):
      self.imgUrlDefaultExtension = self.getSetting('image_download')['default_extension']

    if 'expected_extensions' in self.getSetting('image_download'):
      self.imgExpectedExtensions = self.getSetting('image_download')['expected_extensions']

    list_file = self.getImageList("all")
    
    with open(list_file["path"],'r', encoding=list_file["encoding"]) as file:
        c = csv.reader(file)
        for row in c:
          self.imageDownloadList.append(row)

    self.logger.info("read image list: {} ({})".format(list_file["path"],len(self.imageDownloadList)))
    self._downloadImages(skip_download_if_exists)



  def _downloadImages(self,skip_download_if_exists=True):
    failed = 0
    downloaded = 0
    self.downloaded_list = []

    for idx, row in enumerate(self.imageDownloadList):
      # id, name, image
      url = row[2]
      p = urlparse(url)
      
      if self.imgUrlPathParseRegExp!="" and self.imgUrlPathParseNameIdx >-1:
        l = re.compile(self.imgUrlPathParseRegExp).split(p.path)
        l = list(filter(None, l))
        filename = l[self.imgUrlPathParseNameIdx]
      else:
        filename = os.path.basename(p.path)

      ext = list(reversed(os.path.splitext(filename)))[0].replace(".","").lower()
      has_extension = ext in self.imgExpectedExtensions
    
      if not has_extension:
        if self.imgUrlDefaultExtension=="":
          with urllib.request.urlopen(url) as response:
            info = response.info()
            filename += "." + info.get_content_subtype()
        else:
          filename += "." + self.imgUrlDefaultExtension

      savefile = os.path.join(self.imageFolder,filename)

      try:
        if not os.path.exists(savefile) or skip_download_if_exists==False:
          urllib.request.urlretrieve(url, savefile)
          self.logger.debug("downloaded {} to {}".format(url,savefile))
          downloaded += 1
        else:
          self.logger.debug("skipped download {} to {}".format(url,savefile))
        row.append(filename)
        self.downloaded_list.append(row)
      except:
        self.logger.warning("download failed: {} ({})".format(url,sys.exc_info()[1]))
        failed += 1

      if ((downloaded + failed)!= 0 and (downloaded + failed) % 100 == 0):
          self.logger.info("downloaded: {}, failed: {}".format(downloaded,failed))

    self.logger.info("have {} images".format(len(self.downloaded_list)))

    list_file = self.getImageList("downloaded")

    with open(list_file["path"], 'w+') as file:
      c = csv.writer(file)
      c.writerow(["label","image"])
      for item in self.downloaded_list:
        c.writerow([item[1].encode('utf-8'),item[3]])
    
    self.logger.info("wrote {} ({})".format(list_file["path"], len(self.downloaded_list)))



  def doSaveClassList(self):
    a = pd.DataFrame(np.array(self.downloaded_list))
    self.saveClassList(a)



if __name__ == "__main__":
#  settings_file = "./config/corvidae.yml"
  settings_file = "./config/mnist.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'dwca_reader',logging.INFO)

  reader = imageDownloader(settings,logger)
  reader.downloadImages(skip_download_if_exists=True)
  reader.doSaveClassList()
