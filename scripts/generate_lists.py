import os, csv
import collections
import argparse

parser = argparse.ArgumentParser(description="creates image and class lists for model_trainer.py from a nested image directory.") 
parser.add_argument("--image_root",type=str)
args = parser.parse_args() 

# given a root folder containing subfolders w/ images per class, this script generates 
# the two csv files required for model_trainer.py, downloaded_images.csv and classes.csv

if not args.image_root:
    raise ValueError("need an image root") 
else:
    root_dir = args.image_root


outfile = "downloaded_images.csv"
classfile = "classes.csv"

out = []
classes = []

for root, dirs, files in os.walk(root_dir):
    path = root.split(os.sep)
    for file in files:
        classname = root.replace(root_dir,"")
        out.append([classname, os.path.join(classname,file)])
        classes.append(classname)

freq=collections.Counter(classes)

print("found {} images".format(len(out)))
print("found {} classes".format(len(freq)))

with open(outfile,'w') as csvfile:
    s = csv.writer(csvfile)
    for line in out:
        s.writerow(line)

with open(classfile,'w') as csvfile:
    s = csv.writer(csvfile)
    for item in freq:
        s.writerow([item,freq[item]])

print("wrote {}".format(outfile))
print("wrote {}".format(classfile))