import numpy as np

# read dataset to np_array
cleanData = np.loadtxt("wifi_db/clean_dataset.txt")
noisyDate = np.loadtxt("wifi_db/noisy_dataset.txt")
data_shape = cleanData.shape
attr_count = data_shape[1] - 1


# define the data structure of the decision tree
class Node:
    def __init__(self, attr, val, left, right):
        self.attr = attr
        self.val = val
        self.left = left
        self.right = right

    def show(self):
        if self.left is None and self.right is None:
            return "Then the label is " + str(self.val)
        return "If attribute " + str(self.attr) + " <= " + str(
            self.val) + " {\n  " + self.left.show() + "\n}\n" + "otherwise (attribute " + str(self.attr) + ") > " + str(
            self.val) + " {\n  " + self.right.show()


# count the total number of classes(labels) in the dataset
def count_labels(allData):
    labels = []
    for row in allData:
        l = row[-1]
        if l not in labels:
            labels.append(l)
    return len(labels)


label_count = count_labels(cleanData)


def sortColumn(dataset, attrNum):
    # sort the entire dataset on one attribute
    sortedDataset = dataset[dataset[:,attrNum].argsort()]
    return sortedDataset


