import baseclass
import csv
import os.path
import sys, re
import urllib.request
from urllib.parse import urlparse
import logging
import helpers.logger
import helpers.settings_reader


class dwcaReader(baseclass.baseClass):
  
  """
  DwCA reader:
  - extracts an imagelist from the occurence file from a DwCA archive
  - downloads the images from that list
  
  settings are read from a yaml file
  
  public functions:
    - setClassImageMinimum()
    - readDwca()
    - writeImageList()
    - downloadImages(skip_downloading)
  """
  
  headerCols = []
  imageDownloadList = []

  def __init__(self,project_settings,logger=None):
    if logger is not None:
      self.logger = logger
    self.setSettings(project_settings)
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))


  def setClassImageMinimum(self,image_minimum):
    self.classImageMinimum=image_minimum
    self.logger.info("class image minimum set to {}".format(self.classImageMinimum))


  def readDwca(self):
    self.dwcaInfile = self.getSetting('dwca_infile')
    self.dwcaInfile['path'] = os.path.join(self.projectRoot,self.dwcaInfile['name']) 
    self._getColumnIndices()
    self._extractImageList()


  def writeImageList(self):
    self.imageListFile = self.getSetting('output_lists')['image_list']  
    self.imageListFile['path'] = os.path.join(self.projectRoot,self.imageListFile['file']) 
    self._writeImageList()
  

  def downloadImages(self,skip_downloading=False):    
    # image_url_path_parse
    if 'reg_exp' in self.getSetting('image_url_path_parse'):
      self.imgUrlPathParseRegExp = self.getSetting('image_url_path_parse')['reg_exp']

    if 'name_index' in self.getSetting('image_url_path_parse'):
      self.imgUrlPathParseNameIdx = self.getSetting('image_url_path_parse')['name_index']

    if 'default_extension' in self.getSetting('image_download'):
      self.imgUrlDefaultExtension = self.getSetting('image_download')['default_extension']

    if 'expected_extensions' in self.getSetting('image_download'):
      self.imgExpectedExtensions = self.getSetting('image_download')['expected_extensions']
    
    self._readImageListFile()
    self._downloadImages(skip_downloading)
  
  

  def _getColumnIndices(self):
    for k,v in self.getSetting('dwca_headers').items():
      self.headerCols.append({"name": k,  "columnLabel": v, "index" : -1})

    with open(self.dwcaInfile['path'],'r',encoding=self.dwcaInfile['encoding']) as file:
      c = csv.reader(file)
      row1 = next(c)
    idx=0;
    for header in row1:
      for item in self.headerCols:
        if header==item["columnLabel"]:
          item["index"]=idx
      idx += 1    

    self.logger.debug("resolved DwCA-infile column indices")


  def _extractImageList(self):
    idx_id=-1
    idx_image=-1
    idx_name=-1

    for item in self.headerCols:
      if item["name"]=='id':
        idx_id = item["index"]
      if item["name"]=='images':
        idx_image = item["index"]
      if item["name"]=='name':
        idx_name = item["index"]
        
    if idx_id == -1:
      raise ValueError("couldn't resolve id column index")
    if idx_image == -1:
      raise ValueError("couldn't resolve image column index")
    if idx_name == -1:
      raise ValueError("couldn't resolve label column index")

    self.imageDict={}

    with open(self.dwcaInfile['path'],'r',encoding=self.dwcaInfile['encoding']) as file:
        c = csv.reader(file)
        idx=0
        classes=0
        for row in c:
          if idx==0:
            idx += 1
            continue
          
          if row[idx_name] in self.imageDict:
            self.imageDict[row[idx_name]]["images"] += row[idx_image].split("|")
          else:
            classes += 1
            self.imageDict[row[idx_name]]={
                "label" : row[idx_name],
                "id" : row[idx_id],
                "images": row[idx_image].split("|")
            }

          idx += 1

#    print(self.imageDict)
    self.logger.info("extracted image list ({})".format(classes))


  def _writeImageList(self):
    skipped = 0
    written = 0
    classes = 0
 
    with open(self.imageListFile['path'], 'w+',encoding=self.imageListFile['encoding']) as file:
      c = csv.writer(file)
#      c.writerow(["id","label","image"])
      for x in self.imageDict.items():
        if self.classImageMinimum > 0 and len(x[1]["images"]) < self.classImageMinimum:
          skipped += 1
          continue
#        print(x[0],x[1]["id"],len(x[1]["images"]))
        classes += 1
        for image in x[1]["images"]:
          # id, label, image
          c.writerow([x[1]["id"],x[1]["label"],image])
          written += 1

    self.logger.info("got {} images for {} classes; skipped {} due to image minimum of {}".format(written,classes,skipped,self.classImageMinimum))
    self.logger.info("wrote image list: {}".format(self.imageListFile['path']))



  def _readImageListFile(self):
    with open(self.imageListFile['path'],'r', encoding=self.imageListFile['encoding']) as file:
        c = csv.reader(file)
        for row in c:
          self.imageDownloadList.append(row)

    self.logger.info("read image list: {} ({})".format(self.imageListFile,len(self.imageDownloadList)))



  def _downloadImages(self,skip_downloading=False):    
    if skip_downloading:
      self.logger.warning("BE AWARE! actual downloading is being skipped")

    failed = 0
    downloaded_list = []

    for row in self.imageDownloadList:
      # id, name, image
      url = row[2]
      p = urlparse(url)
      
      if self.imgUrlPathParseRegExp!="" and self.imgUrlPathParseNameIdx >-1:
        l = re.compile(self.imgUrlPathParseRegExp).split(p.path)
        l = list(filter(None, l))
        filename = l[self.imgUrlPathParseNameIdx]
      else:
        filename = os.path.basename(p.path)

      print(filename)

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
        if not skip_downloading:
          urllib.request.urlretrieve(url, savefile)
          self.logger.debug("downloaded {} to {}".format(url,savefile))
        else:
          self.logger.debug("skipped download {} to {}".format(url,savefile))
        row.append(filename)
        downloaded_list.append(row)
      except:
        self.logger.warning("download failed: {} ({})".format(url,sys.exc_info()[1]))
        failed += 1

      if ((len(downloaded_list) + failed) % 100 == 0) and not skip_downloading:
          self.logger.info("downloaded: {}, failed: {}".format(len(downloaded_list),failed))

    self.logger.info("finished downloading: downloaded: {}, failed: {}".format(len(downloaded_list),failed))

    with open(self.downloadedListFile, 'w+') as file:
      c = csv.writer(file)
      c.writerow(["label","image"])
      for item in downloaded_list:
        c.writerow([item[1].encode('utf-8'),item[3]])

    self.logger.info("wrote {} ({})".format(self.downloadedListFile, len(downloaded_list)))

   
    
    
if __name__ == "__main__":
#  settings_file = "./config/martin-collectie.yml"
  settings_file = "./config/corvidae.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)

  reader = dwcaReader(settings,logger)
  reader.readDwca()
  reader.setClassImageMinimum(100)
  reader.writeImageList()
  reader.downloadImages()
