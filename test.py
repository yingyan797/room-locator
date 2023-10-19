import numpy as np
import os
import sys

filepath=clean_dataset.txt
x=[]

for line in open(filepath):
    if line.strip() != "": # handle empty rows in file
            row = line.strip().split(" ")
            x.append(list(map(float, row)))
            
x=np.array(x)
