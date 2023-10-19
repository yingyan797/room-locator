import numpy as np
from dataset import EntropyCalculator, OptimumFinder

cleanData = np.loadtxt("wifi_db/clean_dataset.txt")
noisyDate = np.loadtxt("wifi_db/noisy_dataset.txt")
calculators = []

class Node:
    def __init__(self, attr, val, left, right):
        self.attr = attr
        self.val = val
        self.left = left
        self.right = right


def sortColumn(dataset, attrNum):
    # sort the entire dataset on one attribute
    sortedDataset = dataset[dataset[:,attrNum].argsort()]
    return sortedDataset

def countLabels(allData):
    labels = []
    for row in allData:
        l = row[-1]
        if l not in labels:
            labels.append(l)
    return len(labels)

def createCalculators(allData):
    for attrNum in range(len(allData[0])-1):
        entCalc = EntropyCalculator(countLabels(allData), attrNum)
        entCalc.sortColumn()
        entCalc.groupingSplittingColumn(allData)
        calculators.append(entCalc)
    
def findOptimalSplit(partData):
    # find optimal attribute number and create a calculator for it
    maxAttrEC = None
    maxIG = 0
    rightFinder = None

    for entCalc in calculators:
        rf = entCalc.findOptimum(partData)
        ig = rf.prefixIG()
        if ig > maxIG:
            maxIG = ig
            maxAttrEC = entCalc
            rightFinder = rf

    # find optimal split within optimal attribute
    maxIG = 0
    splitPoint = 0

    leftFinder = OptimumFinder(0, 0, [0 for i in range(maxAttrEC.labelCount)])
    rowBound = 0
    for split in maxAttrEC.splitPoints:
        if split not in range(partData[0][maxAttrEC.attrNum], partData[-1][maxAttrEC.attrNum]):
            continue
        
        value = None
        while True:
            v = partData[rowBound][maxAttrEC.attrNum]
            if v < split:
                if v != value:
                    info = maxAttrEC.valueGroup[v]
                    leftFinder.update(info, 1)
                    rightFinder.update(info, -1)
                    value = v
                rowBound += 1
            else:
                break              

        leftIG = leftFinder.prefixIG()
        rightIG = rightFinder.prefixIG()
        splitIG = (leftIG * leftFinder.valueCount + rightIG * rightFinder.valueCount)/len(partData)
        if splitIG > maxIG:
            maxIG = splitIG
            splitPoint = split
    return splitPoint


def decisionTreeCreating(dataset,depth):
    
    actualnode=Node(None,dataset,None,None)

    allLabels=actualnode.val[:,-1]#Get the column with all the labels
    if np.all(allLabels==allLabels[0]):#Checks if all of the labels are the same
        return (actualnode,depth)
    else:
        split=findOptimalSplit(dataset,0,dataset.shape[0])
        actualnode=Node()
    