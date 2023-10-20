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

    def show(self):
        if self.left is None and self.right is None:
            return "Then the label is "+self.val
        return "If attribute "+self.attr+" <= "+self.val+" {\n  "+self.left.show()+"\n}\n"+"otherwise (attribute "+self.attr+") > "+self.val+" {\n  "+self.left.show()+"\n}\n"


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
    labelCount = countLabels(allData)
    for attrNum in range(len(allData[0])-1):
        entCalc = EntropyCalculator(labelCount, attrNum)
        entCalc.groupingSplittingColumn(sortColumn(allData, attrNum))
        calculators.append(entCalc)
    
def findOptimalSplit(partData):
    # find optimal attribute number and create a calculator for it
    maxAttrEC = None
    maxIG = 0
    rightFinder = None
    for entCalc in calculators:
        dataSorted = sortColumn(partData, entCalc.attrNum)
        rf = entCalc.findOptimum(dataSorted)
        ig = rf.prefixIG()
        if ig > maxIG:
            maxIG = ig
            maxAttrEC = entCalc
            rightFinder = rf
            partData = dataSorted

    # find optimal split within optimal attribute
    maxIG = 0
    rowBound = 0
    splitValue = 0

    leftFinder = OptimumFinder(0, 0, [0 for i in range(maxAttrEC.labelCount)])
    rowNum = 0
    for split in maxAttrEC.splitPoints:
        if split <= partData[0][maxAttrEC.attrNum] or split >= partData[-1][maxAttrEC.attrNum]:
            continue
        
        value = None
        while True:
            v = partData[rowNum][maxAttrEC.attrNum]
            if v < split:
                if v != value:
                    info = maxAttrEC.valueGroup[v]
                    leftFinder.update(info, 1)
                    rightFinder.update(info, -1)
                    value = v
                rowNum += 1
            else:
                break              

        leftIG = leftFinder.prefixIG()
        rightIG = rightFinder.prefixIG()
        splitIG = (leftIG * leftFinder.valueCount + rightIG * rightFinder.valueCount)/len(partData)
        if splitIG > maxIG:
            maxIG = splitIG
            rowBound = rowNum
            splitValue = split
    return rowBound, maxAttrEC.attrNum, splitValue


def decisionTreeCreating(dataset,depth):    
    allLabels = dataset[:,-1]#Get the column with all the labels
    if np.all(allLabels==allLabels[0]):#Checks if all of the labels are the same
        return (Node(None, allLabels[0], None, None),depth)
    else:
        rowBound, attrNum, splitValue = findOptimalSplit(dataset)
        splitNode = Node(attrNum, splitValue, None, None)
        ldata = dataset[:rowBound]
        rdata = dataset[rowBound:]
        splitNode.left, ldepth = decisionTreeCreating(ldata, depth+1)
        splitNode.right, rdepth = decisionTreeCreating(rdata, depth+1)
        return splitNode, max(ldepth, rdepth)

createCalculators(cleanData)
print(findOptimalSplit(cleanData), "\nDecision tree learning:\n")
tree, d = decisionTreeCreating(cleanData, 0)
print(tree.show())    