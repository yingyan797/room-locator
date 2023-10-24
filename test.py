import numpy as np
import os
import sys

filepath='./wifi_db/clean_dataset.txt'
x=[]

for line in open(filepath):
    if line.strip() != "": # handle empty rows in file
            row = line.strip().split(" ")
            x.append(list(map(float, row)))
            
x=np.array(x)

def evaluate(test_dataset,root):
      
    return
