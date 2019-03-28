#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:32:55 2019

@author: maarten
"""

import os

root_dir = '/storage/data/mnist/images/trainingSet/'


f = open("/storage/data/mnist/list.csv", "a")

for root, directories, filenames in os.walk(root_dir):
  for filename in filenames: 
    print (root.replace(root_dir,""),filename)
    f.write(root.replace(root_dir,"")+"\t"+filename+"\n")
    os.rename(
        os.path.join(root,filename),
        os.path.join('/storage/data/mnist/images',filename)
    )

f.close()