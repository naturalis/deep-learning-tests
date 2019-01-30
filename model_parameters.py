#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:56:30 2019

@author: maarten
"""

class modelParameters():

  parameters = {
    "early_stopping" : { "use": True, "monitor": "val_acc", "patience": 3, "verbose": 0, "restore_best_weights": True },
    "save" : { "after_every_epoch": True, "after_every_epoch_monitor": "val_acc", "when_configured": True },
    "model_architecture" : "InceptionV3",
    "model_target_size" : (299,299),
    "minimum_images_per_class" : 50,
    "batch_size" : 32,
    "split" : { "validation": 0.2, "test" : 0.1 }, # test not implemented
    "image_data_generator" : {
      "rotation_range": 90, "shear_range": 0.2, "zoom_range": 0.2,"horizontal_flip": True
      # "width_shift_range": 0.2, "height_shift_range": 0.2, "vertical_flip": True,
    },
    "model_name" : None,
    "training_stages" : None
  }

  def setModelParameters(self,**kwargs):
    for key, value in kwargs.items():
      if key in self.parameters:
        self.parameters[key] = value 
    

  def getModelParameters(self):
    self._setTrainingStagesDefault()
    self._setModelName()
    return self.parameters


  def _setModelName(self):
    self.parameters["model_name"]=self.parameters["model_name"].format(model_architecture=self.parameters["model_architecture"])


  def _setTrainingStagesDefault(self):
    if self.parameters["training_stages"]==None:
      self.parameters["training_stages"]=[
        {
          "label": "top only",
          "frozen_layers": "base_model",
          "loss_function": "categorical_crossentropy",
          "initial_lr": 1e-4,
          "reduce_lr": { "use": True, "monitor": "val_acc", "factor": 0.2, "patience": 2, "min_lr": 1e-8 },
          "epochs": 4,
          "use": True
        },
        {
          "label": "mixed8 and up",
          "frozen_layers": 249,
          "loss_function": "categorical_crossentropy",
          "initial_lr": 1e-4,
          "reduce_lr": { "use": True, "monitor": "val_acc", "factor": 0.1, "patience": 2, "min_lr": 1e-8 },
          "epochs": 200, # 4
          "use": True
        },
        {
          "label": "all",
          "frozen_layers": "none",
          "loss_function": "categorical_crossentropy",
          "initial_lr": 1e-4,
          "reduce_lr": { "use": True, "monitor": "val_acc", "factor": 0.1, "patience": 2, "min_lr": 1e-8 },
          "epochs": 200,
          "use": False
        }
      ]
