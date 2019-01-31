#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:48:51 2018

@author: maarten
"""
import logging, os

class logger:
  
  log_dir = "./"
  log_to_stdout=True
  log_level = logging.INFO

  def __init__(self, logdir, application_name, loglevel=logging.INFO, log_to_stdout=True):
    if logdir is not None:
      self.log_dir = logdir

    if not os.path.exists(self.log_dir):
      os.makedirs(self.log_dir)

    self.log_level = loglevel
    self.logging = logging.getLogger(application_name)
    self.logging.setLevel(self.log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    self.fh = logging.FileHandler(os.path.join(self.log_dir, application_name + '.log'))
    self.fh.setLevel(self.log_level)
    self.fh.setFormatter(formatter)
    self.logging.addHandler(self.fh)
    
    if log_to_stdout:
      self.ch = logging.StreamHandler()
      self.ch.setLevel(self.log_level)
      self.ch.setFormatter(formatter)
      self.logging.addHandler(self.ch)
    else:
      self.log_to_stdout=False
    
  def __del__(self):
    self.fh.close()
    self.logging.removeHandler(self.fh)
    if self.log_to_stdout:
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
