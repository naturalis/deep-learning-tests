import baseclass
import utilities
import csv
import os.path
import logging
import helpers.logger
import helpers.settings_reader

class dwcaReader(baseclass.baseClass):
  
  """
  DwCA reader:
  - extracts an imagelist from the occurence file from a DwCA archive
  """
  
  headerCols = []
  imageDownloadList = []
  downloadedListFile = []
  classImageMinimum = 2

  def __init__(self,project_settings,logger):
    baseclass.baseClass.__init__(self)
    self.logger = logger
    self.setSettings(project_settings)
    self._setClassImageMinimum()
    self.logger.info("project: {}; program: {}".format(self.projectName,self.__class__.__name__))
    

  def _setClassImageMinimum(self):
    try:
      self.classImageMinimum = self.getSetting("images_per_class_minimum")
    except ValueError:
      pass
    self.logger.info("class image minimum set to {}".format(self.classImageMinimum))


  def readDwca(self):
    self.dwcaInfile = self.getSetting('dwca_infile')
    self.dwcaInfile['path'] = os.path.join(self.projectRoot,self.dwcaInfile['file']) 
   
    # get column indices
    for k,v in self.getSetting('dwca_infile')['headers'].items():
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
    self._extractImageList()


  def saveImageList(self):
    skipped = 0
    written = 0
    classes = 0
    
    list_file = self.getImageList("all")
      
    with open(list_file["path"], 'w+') as file:
      c = csv.writer(file)
#      c.writerow(["id","label","image"])
      for x in self.imageDict.items():
        if self.classImageMinimum > 0 and len(x[1]["images"]) < self.classImageMinimum:
          skipped += 1
          continue
        # print(x[0],x[1]["id"],len(x[1]["images"]))
        classes += 1
        for image in x[1]["images"]:
          # id, label, image
          c.writerow([x[1]["id"],x[1]["label"].encode('utf-8'),image])
          written += 1

    self.logger.info("got {} images for {} classes; skipped {} due to image minimum of {}".format(written,classes,skipped,self.classImageMinimum))
    self.logger.info("wrote image list: {}".format(list_file["path"]))
  

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



if __name__ == "__main__":
  settings_file=utilities.utilities.getSettingsFilePath()
  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'dwca_reader',logging.INFO)

  reader = dwcaReader(settings,logger)
  reader.readDwca()
  reader.saveImageList()
