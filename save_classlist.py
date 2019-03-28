#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 09:54:07 2019

@author: maarten.schermer@naturalis.nl
"""

import os
import logging
import helpers.logger
import helpers.settings_reader
import utilities


if __name__ == "__main__":
#  settings_file = "./config/corvidae.yml"
  settings_file = "./config/mnist.yml"

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)

  reader = utilities.utilities(settings,logger)
  reader.doSaveClassList()
