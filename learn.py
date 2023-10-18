import numpy as np
from dataset import EntropyCalculator

cleanData = np.loadtxt("wifi_db/clean_dataset.txt")
noisyDate = np.loadtxt("wifi_db/noisy_dataset.txt")

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

def findOptimalSplit(dataset):
    # find optimal attribute number and create a calculator for it
    maxAttrEC = None
    maxInformationGain = 0
    for attrNum in range(len(dataset[0])-1):
        entCalc = EntropyCalculator(dataset, countLabels(dataset), attrNum)
        entCalc.sortColumn()
        entCalc.groupingSplittingColumn()
        ig = entCalc.colRangeIG(None)[0]
        if ig > maxInformationGain:
            maxInformationGain = ig
            maxAttrEC = entCalc

    # find optimal split within optimal attribute
    maxAttrEC.sortColumn()
    startValue = maxAttrEC.dataset[0][maxAttrEC.attrNum]
    endValue = maxAttrEC.dataset[-1][maxAttrEC.attrNum]
    maxInformationGain = 0
    splitPoint = 0
    for split in maxAttrEC.splitPoints:
        leftIG, leftCount = maxAttrEC.colRangeIG((startValue, split))
        rightIG, rightCount = maxAttrEC.colRangeIG(split, endValue)
        splitInformationGain = (leftIG * leftCount + rightCount * rightCount)/len(dataset)
        if splitInformationGain > maxInformationGain:
            maxInformationGain = splitInformationGain
            splitPoint = split
    return splitPoint

    
def decisionTreeLearning():
    return



