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
            return "Then the label is "+str(self.val)
        return "If attribute "+str(self.attr)+" <= "+str(self.val)+" {\n  "+self.left.show()+"\n}\n"+"otherwise (attribute "+str(self.attr)+") > "+str(self.val)+" {\n  "+self.right.show()


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
        entCalc.groupSplitValues(sortColumn(allData, attrNum))
        calculators.append(entCalc)
    
def findOptimalSplit(partData):
    # find optimal attribute number and create a calculator for it
    maxIG = 0
    maxAttr = 0
    splitValue = 0
    for entCalc in calculators:
        attr = entCalc.attrNum
        dataSorted = sortColumn(partData, attr)
        rightFinder = entCalc.findOptimum(dataSorted)
        leftFinder = OptimumFinder(0, 0, [0 for i in range(entCalc.labelCount)])
        rowNum = 0
        for split in entCalc.splitPoints:
            if split <= partData[0][attr] or split > partData[-1][attr]:
                continue
            value = None
            while True:
                v = partData[rowNum][attr]
                if v < split:
                    if v != value:
                        info = entCalc.valueGroup[v]
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
                maxAttr = attr
                splitValue = split
    
    return maxAttr, splitValue

def getSplitedDatasets(dataset, attr, split):#Get the left and right datasets from a splitting point
        left_set=dataset[dataset[:,attr] < split]
        right_set=dataset[dataset[:,attr] >= split]
        return (np.array(left_set),np.array(right_set))

def decisionTreeCreating(dataset,depth):  
    allLabels = dataset[:,-1]#Get the column with all the labels
    if len(allLabels) == 0:
        return Node(None, 999, None, None),999
    if np.all(allLabels==allLabels[0]) or depth >= 100:#Checks if all of the labels are the same
        return (Node(None, allLabels[0], None, None),depth)
    else:
        attrNum, splitValue = findOptimalSplit(dataset)
        splitNode = Node(attrNum, splitValue, None, None)
        ldata, rdata = getSplitedDatasets(dataset, attrNum, splitValue)
        splitNode.left, ldepth = decisionTreeCreating(ldata, depth+1)
        splitNode.right, rdepth = decisionTreeCreating(rdata, depth+1)
        return splitNode, max(ldepth, rdepth)

createCalculators(cleanData)
print(findOptimalSplit(cleanData), "\nDecision tree learning:\n")
tree, d = decisionTreeCreating(cleanData, 0)
print(tree.show())    