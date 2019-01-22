#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:48:51 2018

@author: maarten
"""
import logging, os

class logger:
  
  log_dir = "./"

  def __init__(self, logdir, application_name, loglevel=logging.INFO):
    if logdir is not None:
      self.log_dir = logdir

    if not os.path.exists(self.log_dir):
      os.makedirs(self.log_dir)
      
    self.logging = logging.getLogger(application_name)
    self.logging.setLevel(loglevel)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    self.fh = logging.FileHandler(os.path.join(self.log_dir, application_name + '.log'))
    self.fh.setLevel(loglevel)
    self.fh.setFormatter(formatter)
    self.logging.addHandler(self.fh)

    self.ch = logging.StreamHandler()
    self.ch.setLevel(loglevel)
    self.ch.setFormatter(formatter)
    self.logging.addHandler(self.ch)
    
  def __del__(self):
    self.fh.close()
    self.logging.removeHandler(self.fh)
    self.ch.close()
    self.logging.removeHandler(self.ch)

  def debug(self,msg):
    self.logging.debug(msg)
  
  def info(self,msg):
    pass
    self.logging.info(msg)
        
  def warning(self,msg):
    self.logging.warning(msg)

  def error(self,msg):
    self.logging.error(msg)

  def critical(self,msg):
    self.logging.critical(msg)
