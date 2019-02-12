#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten
"""
import baseclass
import logging
import helpers.logger
import helpers.settings_reader
import model_parameters
import os
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions

#import numpy as np



#from keras.preprocessing import image
#from keras import optimizers

# https://datascience.stackexchange.com/questions/32814/error-while-using-flow-from-generator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class modelTest(baseclass.baseClass):

  def __init__(self,project_settings, logger=None):
    if logger is not None:
      self.logger = logger
    self.settings = project_settings
    self.setProjectRoot(project_settings)
    self.setModelRepoFolder(project_settings)
    self.setClassListPath(project_settings)
    self.setTestImageFolder(project_settings)
    self.imageListImageCol = self.settings['output_lists']['image_list']['col_image']


  def setTestSettings(self, model_architecture, model_target_size):
    self.setModelArchitecture(model_architecture)
    self.setModelTargetSize(model_target_size)


  def setModelParameters(self,parameters):    
    self.setTestSettings(
        model_architecture = parameters["model_architecture"],
        model_target_size = parameters["model_target_size"],
    )

  def readTestImagesFromDirectory(self):
    self.testFiles = [f for f in os.listdir(self.testImageFolder) if os.path.isfile(os.path.join(self.testImageFolder, f))]
    self.testdf = pd.DataFrame({self.imageListImageCol:self.testFiles})
    self.logger.debug("read test dir {}, found {} image(s)".format(self.testImageFolder,len(self.testFiles)))
    # print(self.testdf)
    
    
  def fuck1(self):
    
    for x,y in self.testdf.iterrows():
      testImage = os.path.join(self.testImageFolder,y["image"])
      img = image.load_img(testImage, target_size=self.modelTargetSize)
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      preds = self.model.predict(x)
      index = np.argmax(preds[0])
      certainty = np.max(preds[0])
      this_class = self.resolveClassId(index)

      print("{}: {} ({})".format(y["image"], this_class, certainty))
    
    
    
  def initTestGenerator(self):


    test_datagen=ImageDataGenerator(rescale=1./255.)

    self.test_generator=test_datagen.flow_from_dataframe(
        dataframe=self.testdf, 
        directory=self.testImageFolder,
        x_col=self.imageListImageCol, 
        y_col=None, 
        class_mode=None, 
        target_size=self.modelTargetSize,
        batch_size=1,
        shuffle=False)

    print(self.testImageFolder)
    return

    for i in self.test_generator:
        idx = (self.test_generator.batch_index - 1) * self.test_generator.batch_size
        print(self.test_generator.filenames[idx : idx + self.test_generator.batch_size])
        
    self.test_generator.reset()


  def doTestImagesPredictions(self):




    self.test_generator.reset()

    step_size_train=self.test_generator.n//self.test_generator.batch_size
    image_predictions=self.model.predict_generator(self.test_generator,steps=step_size_train, verbose=1)

    
#    val_generator = datagen.flow_from_directory(
#        path+'/valid',
#        target_size=(224, 224),
#        batch_size=batch_size,)
#
#x,y = val_generator.next()
#for i in range(0,1):
#    image = x[i]
#    plt.imshow(image.transpose(2,1,0))
#    plt.show()
    
    for image_prediction in image_predictions:
      index = np.argmax(image_prediction)
      certainty = np.max(image_prediction)
      this_class = self.resolveClassId(index)
      
#      print("{}: {} ({})".format(testImage.replace(self.testImageRootFolder,""),this_class, certainty))


#    preds = self.model.predict(x)
#    index = np.argmax(preds[0])
#    certainty = np.max(preds[0])
#    
#
#    print("{}: {} ({})".format(testImage.replace(self.testImageRootFolder,""), 
#          this_class, 
#          certainty))
    



if __name__ == "__main__":
  settings_file = "./config/martin-collectie.yml"
#  settings_file = "./config/aliens.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'test',logging.DEBUG)
  params=model_parameters.modelParameters()

  params.setModelParameters(
      # early_stopping={ "use": True, "monitor": "val_acc", "patience": 3, "verbose": 0, "restore_best_weights": True},
      # save={ "after_every_epoch": True, "after_every_epoch_monitor": "val_acc", "when_configured": True },
  )

  test = modelTest(settings, logger)
  test.setModelParameters(params.getModelParameters())
  test.setModelName("martin_InceptionV3")
  test.loadModel()

  test.readTestImagesFromDirectory()
  test.readClasses()
  test.fuck1()
#  test.initTestGenerator()
#  test.doTestImagesPredictions()

#  test.runPredictions()



# confusion matrix
  