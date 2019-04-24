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
  settings_file=utilities.utilities.getSettingsFilePath()

  settings=helpers.settings_reader.settingsReader(settings_file).getSettings()
  logger = helpers.logger.logger(os.path.join(settings["project_root"] + settings["log_folder"]),'training',logging.INFO)

  utils = utilities.utilities(settings,logger)
  utils.doSaveClassList()
