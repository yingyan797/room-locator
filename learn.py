import numpy as np
from dataset import EntropyCalculator

cleanData = np.loadtxt("wifi_db/clean_dataset.txt")
noisyDate = np.loadtxt("wifi_db/noisy_dataset.txt")
calculators = []

class Node:
    def __init__(self, attr, val, left, right):
        self.attr = attr
        self.val = val
        self.left = left
        self.right = right

def countLabels(dataset):
    labels = []
    for row in dataset:
        l = row[-1]
        if l not in labels:
            labels.append(l)
    return len(labels)

def createCalculators(dataset):
    for attrNum in range(len(dataset[0])-1):
        entCalc = EntropyCalculator(dataset, countLabels(dataset), attrNum)
        entCalc.sortColumn()
        entCalc.groupingSplittingColumn()
        calculators.append(entCalc)
    
def findOptimalSplit(dataset, fromRow, untilRow):
    # find optimal attribute number and create a calculator for it
    maxAttrEC = None
    maxIG = 0

    for entCalc in calculators:
        ig = entCalc.colRangeIG(fromRow, untilRow)
        if ig > maxIG:
            maxIG = ig
            maxAttrEC = entCalc

    # find optimal split within optimal attribute
    maxIG = 0
    splitPoint = 0
    for split in maxAttrEC.splitPoints:
        leftIG, leftCount = maxAttrEC.colRangeIG((startValue, split))
        rightIG, rightCount = maxAttrEC.colRangeIG(split, endValue)
        splitIG = (leftIG * leftCount + rightIG * rightCount)/len(dataset)
        if splitIG > maxIG:
            maxIG = splitIG
            splitPoint = split
    return splitPoint

    
def decisionTreeLearning(dataset, rowRange, depth):
    return



