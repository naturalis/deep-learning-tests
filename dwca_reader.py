import csv
import os.path
import sys, re
import urllib.request
from urllib.parse import urlparse
import logging
import helpers.logger
import helpers.settings_reader
import pandas as pd

class header:
  name = ""
  columnLabel = ""
  index = -1

  def __init__(self,name,columnLabel,):
    self.name = name
    self.columnLabel = columnLabel

  def setIndex(self,idx):
    self.index = idx


class dwcaReader:
  
  """
  DwCA reader:
  - extracts an imagelist from the occurence file from a DwCA archive
  - downloads the images from that list
  
  settings are read from a yaml file
  
  public functions:
    - extractImageList()
      extracts imagelist from DwCA file specified in the settings-file
    - writeImageList(outfile)
      writes imagelist to file (per line: id, classname, image)
    - readImageDownloadList()
      reads list of images to download from file (expects: id, classname, image)
    - downloadImages(skip_downloading)
      download the images on the list read by readImageDownloadList()
    
  TO DO:
    - move names of in and outfiles to YAML as well?
  """
  
  project_root = "./"
  imageListFile = "imagelist.csv"
  downloadedListFile = "downloaded.csv"
  infile = ""
  infileEncoding = 'utf-8-sig'
  headerCols = []
  imageList = []
  imageDownloadList = []
  imgUrlPathParseRegExp = ""
  imgUrlPathParseNameIdx = -1
  imgUrlDefaultExtension = ""
  imgExpectedExtensions = []
  imgDownloadFolder = ""
  classImageMinimum = 0


  def __init__(self,settings,logger=None):
    self.settings = settings
    if logger is not None:
      self.logger = logger
    self._readSettings()
    self._getColumnIndices()
    self._createDownloadFolder()


  def _readSettings(self):
    # print(self.settings)/storage/data/martin-collectie/tests/
    self.project_root = self.settings['project_root']

    # dwca_infile
    self.infile = os.path.join(self.project_root ,self.settings['dwca_infile']['name'])
    if 'encoding' in self.settings['dwca_infile']:
      self.infileEncoding = self.settings['dwca_infile']['encoding']
    
    # dwca_headers
    for k,v in self.settings['dwca_headers'].items():
      self.headerCols.append(header(k,v))

    # image_url_path_parse
    if 'reg_exp' in self.settings['image_url_path_parse']:
      self.imgUrlPathParseRegExp = self.settings['image_url_path_parse']['reg_exp']
    if 'name_index' in self.settings['image_url_path_parse']:
      self.imgUrlPathParseNameIdx = self.settings['image_url_path_parse']['name_index']

    # image_download
    if 'folder' in self.settings['image_download']:
      self.imgDownloadFolder = os.path.join(self.project_root, self.settings['image_download']['folder'])
    if 'default_extension' in self.settings['image_download']:
      self.imgUrlDefaultExtension = self.settings['image_download']['default_extension']
    if 'expected_extensions' in self.settings['image_download']:
      self.imgExpectedExtensions = self.settings['image_download']['expected_extensions']

    # output_lists
    if 'image_list' in self.settings['output_lists']:
      self.imageListFile = os.path.join(self.project_root, self.settings['output_lists']['image_list']['file'])
      self.imageListFileEncoding = self.settings['output_lists']['image_list']['encoding']

    if 'downloaded_list' in self.settings['output_lists']:
      self.downloadedListFile = os.path.join(self.project_root, self.settings['output_lists']['downloaded_list'])


  def _getColumnIndices(self):
    with open(self.infile,'r',encoding=self.infileEncoding) as file:
      c = csv.reader(file)
      row1 = next(c)
    idx=0;
    for header in row1:
      for item in self.headerCols:
        if header==item.columnLabel:
          item.setIndex(idx)
      idx += 1


  def _createDownloadFolder(self):
    if not os.path.exists(self.imgDownloadFolder):
      os.makedirs(self.imgDownloadFolder)
      self.logger.debug("created download folder: {}".format(self.imgDownloadFolder))


  def extractImageList(self):
    for item in self.headerCols:
      if item.name=='id':
        idx_id = item.index
      if item.name=='images':
        idx_image = item.index
      if item.name=='name':
        idx_name = item.index

    self.imageDict={}

    with open(self.infile,'r',encoding=self.infileEncoding) as file:
        c = csv.reader(file)
        idx=0
        for row in c:
          if idx==0:
            idx += 1
            continue
          
          if row[idx_name] in self.imageDict:
            self.imageDict[row[idx_name]]["images"] += row[idx_image].split("|")
          else:
            self.imageDict[row[idx_name]]={
                "label" : row[idx_name],
                "id" : row[idx_id],
                "images": row[idx_image].split("|")
            }

          idx += 1

#    print(self.imageDict)
    self.logger.info("extracted image list ({})".format(idx-1))


  def setClassImageMinimum(self,image_minimum):
    self.classImageMinimum=image_minimum
    self.logger.info("class image minimum set to {}".format(self.classImageMinimum))


  def writeImageList(self):
    skipped = 0
    written = 0
    classes = 0
    with open(self.imageListFile, 'w+',encoding=self.imageListFileEncoding) as file:
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

    self.logger.info("got {} images for {} classes; skipped {} due to image minimum {}".format(written,classes,skipped,self.classImageMinimum))
    self.logger.info("wrote image list: {}".format(self.imageListFile))


  def _readImageListFile(self):
    with open(self.imageListFile,'r', encoding=self.infileEncoding) as file:
        c = csv.reader(file)
        for row in c:
          self.imageDownloadList.append(row)

    self.logger.info("read image list: {} ({})".format(self.imageListFile,len(self.imageDownloadList)))


  def downloadImages(self,skip_actual_downloading=False):    
    self._readImageListFile()
    
    if skip_actual_downloading:
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

      savefile = os.path.join(self.imgDownloadFolder,filename)

      try:
        if not skip_actual_downloading:
          urllib.request.urlretrieve(url, savefile)
          self.logger.debug("downloaded {} to {}".format(url,savefile))
        else:
          self.logger.debug("skipped download {} to {}".format(url,savefile))
        row.append(filename)
        downloaded_list.append(row)
      except:
        self.logger.warning("download failed: {} ({})".format(url,sys.exc_info()[1]))
        failed += 1

      if ((len(downloaded_list) + failed) % 100 == 0) and not skip_actual_downloading:
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
  reader.extractImageList()
  reader.setClassImageMinimum(100)
  reader.writeImageList()
  reader.downloadImages()



