import numpy as np

class ValueInfo:
    # storing the frequency count of each label appeared in one value, total count, and whether splitting criteria is met
    def __init__(self, labelCount, count=0):
        # label count is 4 in this dataset, room 1,2,3,4 
        self.frequency = [0 for i in range(labelCount)]
        self.count = count
        self.unique = None

    def unique_label(self):
        if self.unique != None:
            return self.unique
        num_distinct = 0
        label = -1
        for i in range(len(self.frequency)):
            if self.frequency[i] > 0:
                num_distinct += 1
                label = i + 1
            if num_distinct > 1:
                self.unique = -1
                return self.unique
        self.unique = label
        return self.unique


class OptimumFinder:    # the object storing the cumulative sum of label frequency and calculating entropy for each value
    def __init__(self, size, prefix_freq):
        self.size = size
        self.prefix_freq = prefix_freq

    def entropy(self):  # calculate the current value's entropy
        ent = 0
        for pf in self.prefix_freq:
            if pf > 0:
                prob = pf / self.size
                ent -= prob * np.log2(prob)
        return ent

    def update(self, info, fac):    # Adding and deducting value info in finding optimum splits
        for i in range(len(self.prefix_freq)):
            self.prefix_freq[i] += fac * info.frequency[i]
        self.size += fac * info.count
