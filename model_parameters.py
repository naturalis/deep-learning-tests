#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 14:56:30 2019

@author: maarten
"""

import baseclass

class modelParameters(baseclass.baseClass):
  

  def __init__(self,settings,logger=None):
    if logger is not None:
      self.logger = logger
    self.settings = settings

 
    self.params["model_name"]
    self.params["model_architecture"]
    self.params["input_size"]


    self.params["validation_split"]
    self.params["batch_size"]
    self.params["epochs"]
    self.params["initial_lr"]
    self.params["reduce_lr"]
    self.params["early_stopping"]
    self.params["save"]

   
    self.params["model"]
    self.params["add_extra_toppings"]
    