import numpy as np

class EntropyCalculator:
    def __init__(self, dataset, labelCount):
        self.dataset = dataset
        self.labelCount = labelCount
        self.valueGroup = {}
        self.splitPoints = []

    def sortColumn(self):
        return
    
    def groupingSplittingColumn(self, attrNum):
        self.valueGroup = {}
        val = self.dataset[0][attrNum]
        label = self.dataset[0][-1]
        labelFreq = [0 for i in range(self.labelCount)]
        labelFreq[label-1] = 1
        self.valueGroup[val] = labelFreq
        for row in self.dataset[1:]:
            v = row[attrNum]
            l = row[-1]
            if v in self.valueGroup:
                self.valueGroup[v][l-1] += 1
            else:
                labelFreq = [0 for i in range(self.labelCount)]
                labelFreq[l-1] = 1
                self.valueGroup[v] = labelFreq
            if l != label:
                split = v
                if split != val:
                    split = (split+val)/2
                if split not in self.splitPoints:
                    self.splitPoints.append(split)
                label = l
            val = v

    def valueGroupEntropy(self, labelFreq):
        s = sum(labelFreq)
        ent = 0
        for f in labelFreq:
            if f > 0:
                prob = f/s
                print(prob)
                ent -= prob * np.log2(prob)
        return ent
    
    def columnRangeInformationGain(self, fromVal, toVal):
        return


d = np.array([[0,3], [1,1], [2,1], [2,2]])
ec = EntropyCalculator(d, 4)
ec.groupColumn(0)
