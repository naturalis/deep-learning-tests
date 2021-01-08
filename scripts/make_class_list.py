import os, csv
import collections
import argparse

parser = argparse.ArgumentParser(description="bla") 
parser.add_argument("--image_file",type=str)
parser.add_argument("--class_file_name",type=str)
args = parser.parse_args() 

if not args.image_file:
    raise ValueError("need an image file") 
else:
    image_file = args.image_file

if not args.class_file_name:
    raise ValueError("need a class file name") 
else:
    classfile = args.class_file_name

classes = []

with open(image_file) as csvfile:
    s = csv.reader(csvfile)
    for row in s:
        classes.append(row[0])

freq=collections.Counter(classes)

print("found {} classes".format(len(freq)))

with open(classfile,'w') as csvfile:
    s = csv.writer(csvfile)
    for item in freq:
        s.writerow([item,freq[item]])

print("wrote {}".format(classfile))