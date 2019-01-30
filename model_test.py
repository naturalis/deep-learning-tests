#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten
"""
import baseclass
import logging
from keras_preprocessing.image import ImageDataGenerator
import pandas as pd
import helpers.logger
import helpers.settings_reader
import numpy as np
from os import listdir
from os.path import isfile, join
from keras.preprocessing import image


# https://datascience.stackexchange.com/questions/32814/error-while-using-flow-from-generator
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class modelTest(baseclass.baseClass):

  def __init__(self,settings,logger=None):
    if logger is not None:
      self.logger = logger
    self.settings = settings
    self.setProjectRoot(settings)
    self.setModelRepoFolder(settings)
    self.setClassListPath(settings)
    self.setTestImageFolder(settings)
    self.imageListImageCol = self.settings['output_lists']['image_list']['col_image']
    

  def setSettings(self, **kwargs):
    if 'input_size' in kwargs:
      self.setModelInputSize(kwargs['input_size'])
    else:
      if self.modelArchitecture == "InceptionV3" or self.modelArchitecture == "Xception":
        self.setModelInputSize((299,299,3))
      elif self.modelArchitecture == "VGG16":
        self.setModelInputSize((224,224,3))
      else:
        raise ValueError("no input size set")



  def readTestImages(self):
    self.testFiles = [f for f in listdir(self.testImageFolder) if isfile(join(self.testImageFolder, f))]
    self.testdf = pd.DataFrame({self.imageListImageCol:self.testFiles})
    # print(self.testdf)
    self.logger.debug("read test dir {}, found {} image(s)".format(self.testImageFolder,len(self.testFiles)))


  def runPredictions(self):

    datagen=ImageDataGenerator(preprocessing_function=self.preProcess)

    test_generator=datagen.flow_from_dataframe(
        dataframe=self.testdf, 
        directory=self.testImageFolder,
        x_col=self.imageListImageCol, 
        y_col=None, 
        class_mode=None, 
        shuffle=False,
        target_size=self.modelTargetSize,
        batch_size=2)
    
    #    should be done on test, or on validation if no test exists
    score = self.model.evaluate_generator(
        test_generator,
        test_generator.n/test_generator.batch_size,
        verbose=1)

    self.logger.info("score: {}".format(set(zip(self.model.metrics_names,score))))
    
    return


    
    
    
    testImage=join(self.testImageFolder,self.testFiles[0])
    img = image.load_img(testImage, target_size=self.modelTargetSize)
    x1 = image.img_to_array(img)
    x1 = np.expand_dims(x1, axis=0)
    
    pred = self.model.predict_classes(x1, batch_size=1, verbose=1)
    print(   np.argmax(pred,axis=0))
    
    return
    
    
    import keras.applications.inception_v3

    testImage=join(self.testImageFolder,self.testFiles[0])
    img = image.load_img(testImage, target_size=self.modelTargetSize)
    x1 = image.img_to_array(img)

    testImage=join(self.testImageFolder,self.testFiles[1])
    img = image.load_img(testImage, target_size=self.modelTargetSize)
    x2 = image.img_to_array(img)
    
    x1 = np.expand_dims(x1, axis=0)
    x1 = keras.applications.inception_v3.preprocess_input(x1)
    x2 = np.expand_dims(x2, axis=0)
    x2 = keras.applications.inception_v3.preprocess_input(x2)
    
    p = self.model.predict_on_batch(x1,x2)
    print(testImage,p)
    
    return
    
    
    
    datagen=ImageDataGenerator(preprocessing_function=self.preProcess)
   
    
    print(self.modelTargetSize)
    
    test_generator=datagen.flow_from_dataframe(
        dataframe=self.testdf, 
        directory=self.testImageFolder,
        x_col=self.imageListImageCol, 
        y_col=None, 
        class_mode=None, 
        shuffle=False,
        target_size=self.modelTargetSize,
        batch_size=2)
    
    
    predict = self.model.predict_generator(
        test_generator, 
        steps=2,
        max_queue_size=10, 
        workers=1, 
        use_multiprocessing=False, 
        verbose=1)

    # predict_generator takes your test data and gives you the output
    print(predict)
    print(len(predict))
      
    
#    predict_generator takes your test data and gives you the output.
#    evaluate_generator uses both your test input and output. It first predicts output using training input and then evaluates performance by comparing it against your test output. So it gives out a measure of performance, i.e. accuracy in your case.
#    score = self.model.evaluate_generator(test_generator,steps=4)

#    self.logger.info("score: {}".format(set(zip(self.model.metrics_names,score))))
#
#    predictions = self.model.predict_generator(
#        test_generator,iii
#        test_generator.n/test_generator.batch_size,
#        verbose=1,
#        steps=4)
#    
#    bla = np.argmax(predictions,axis=1)
#    print(bla.shape)
#    print(bla)
    
    
#    print(predictions)
#    print(type(predictions))
    
#    p = self.model.predict()
    
  
#    print(self.traindf.groupby(by="label").count().sort_values(by="image",ascending=False))
#    
#    bla = validation_generator.class_indices
#    index_to_class = dict(zip(bla.values(),bla.keys()))
#    print(index_to_class[222])








if __name__ == "__main__":
  #  settings_file = "./config/martin-collectie.yml"
  settings_file = "./config/aliens.yml"
  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger=helpers.logger.logger(join(settings["project_root"] + settings["log_folder"]),settings["project_name"] + '/model_test',logging.DEBUG)

  test = modelTest(settings, logger)
  test.setModelName("alien_predator_inception_2")
  test.setModelArchitecture("InceptionV3") # Inception, Xception, VGG16
  test.setSettings()
  test.loadModel()
  test.readTestImages()
  test.runPredictions()

  
