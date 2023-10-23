import numpy as np

class ValueInfo:
    # storing the frequency count of each label appears in one value, as well as its entropy
    # "value" takes the form of x1 = 61, 60, 75 etc. equivalent to "sunny", "windy" in tennis example
    def __init__(self, labelCount, count=0, entropy=0):
        # label count is 4 in this dataset, room 1,2,3,4 
        self.labelFreq = [0 for i in range(labelCount)]
        self.count = count
        self.entropy = entropy
        self.pure = None

    def isPure(self):
        if self.pure != None:
            return self.pure
        numDistinct = 0
        label = None
        for i in range(len(self.labelFreq)):
            if self.labelFreq[i] > 0:
                numDistinct += 1
                label = i+1
            if numDistinct > 1:
                return -1
        return label
    
    def valueEntropy(self):
        self.count = sum(self.labelFreq)
        ent = 0
        for f in self.labelFreq:
            if f > 0:
                prob = f/self.count
                ent -= prob * np.log2(prob)
        self.entropy = ent
        return ent
    
    def show(self):
        # for testing debugging only
        print(self.labelFreq, self.entropy)

# Each entropy calculator is corresponded to one attribute and each calculates the entropy of that attribute
class EntropyCalculator:
    def __init__(self, labelCount, attrNum):
        self.attrNum = attrNum
        self.labelCount = labelCount    # in this dataset is 4
        self.valueGroup = {}            # value: ValueInfo (dictionary)
        self.splitPoints = []           # which value needs splitting (contains different labels)
          
    def groupSplitValues(self, allData):
        # create valueGroup dictionary and find the splitting points in one loop
        val = None
        for row in allData:
            v = row[self.attrNum]    # value
            l = int(row[-1])         # label
            if v == val:
                self.valueGroup[v].labelFreq[l-1] += 1
            else:
                info = ValueInfo(self.labelCount)
                info.labelFreq[l-1] = 1
                self.valueGroup[v] = info
                if val != None:
                    self.valueGroup[val].valueEntropy()
                val = v
        
        valsInfo = list(self.valueGroup.items())
        i = 0
        while i < len(valsInfo)-1:
            if valsInfo[i][1].isPure() < 0 or valsInfo[i+1][1].isPure() != valsInfo[i][1].isPure():
                self.splitPoints.append(valsInfo[i+1][0])
            i += 1
    
    def findOptimum(self, partData):
        # include fromRow, exclude untilRow
        # find the information gain on a subset of values (esplitted)
        attrEntSum = 0
        totFreq = [0 for i in range(self.labelCount)]

        value = None
        for row in partData:
            v = row[self.attrNum]
            if value != v:
                info = self.valueGroup[v]
                attrEntSum += info.count * info.entropy
                for i in range(self.labelCount):
                    totFreq[i] += info.labelFreq[i]
                value = v

        return OptimumFinder(attrEntSum, sum(totFreq), totFreq)
        
class OptimumFinder:
    def __init__(self, entropySum, valueCount, labelFreq):
        self.entropySum = entropySum
        self.valueCount = valueCount
        self.labelFreq = labelFreq

    def prefixIG(self):
        if self.entropySum == 0:
            return 0
        totEntropy = 0
        for tf in self.labelFreq:
            if tf > 0:
                prob = tf/self.valueCount
                totEntropy -= prob * np.log2(prob)

        # IG = H(D) - sum(HD'attr)
        return totEntropy - self.entropySum/self.valueCount
    
    def update(self, info, fac):
        self.entropySum += fac * info.count * info.entropy
        for i in range(len(self.labelFreq)):
            self.labelFreq[i] += fac*info.labelFreq[i]
        self.valueCount += fac * info.count



# test value entropy

# d = np.array([[0,3], [1,1], [2,1], [2,2]])
# ec = EntropyCalculator(d, 4, 0)
# ec.groupingSplittingColumn()
# for v in ec.valueGroup.values():
#     v.show()
# print(ec.colRangeIG(0,3))

