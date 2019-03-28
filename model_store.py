#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 09:32:19 2018

@author: maarten
"""

import hashlib
import time

class modelStore:

  def __init__(self):
    pass

  def getVersionNumber(self):
    m = hashlib.md5()
    m.update(str(round(time.time())).encode("utf-8-sig"))
    return m.hexdigest()


  def storeModelPackage(self):
    pass
#    model
#    params
#    results
#    datetime
#    dataset
#     files + original URL
#     split
#   short description
#   notes
#   results



if __name__ == "__main__":
  x=modelStore()
  print(x.getVersionNumber())
  
