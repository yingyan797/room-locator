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
    sortedDataset = dataset[dataset[:, attrNum].argsort()]
    return sortedDataset


def find_optimal_split_point(dataset):
    max_IG = -1  # This is the Information gained
    max_IG_attr = -1  # This is the attribute where the max IG value belongs
    max_IG_split_value = -1  # This is the split value where the max IG value appeared at
    for column in range(attr_count):
        sorted_dataset = sortColumn(dataset, column)  # sort the dataset
        label_frequency = [0 for i in range(label_count)]
        row_value = None
        for row in sorted_dataset:
            label = int(row[-1])
            label_frequency[label - 1] += 1  # calculate the frequency of this label
            if not row_value:
                row_value = row[column]
            elif row_value != row[column]:
                row_value = row[column]

    #TODO: Complete the function


def decision_tree_learning(dataset, depth):
    allLabels = dataset[:, -1]  # Get the column with all the labels
    if len(allLabels) == 0:
        return Node(None, 999, None, None), 999
    if np.all(allLabels == allLabels[0]) or depth >= 100:  # Checks if all of the labels are the same
        return Node(None, allLabels[0], None, None), depth
    # TODO: Find optimal spliting point, Store spliting point as node, Split the data into 2 datasets, repeat until
    #  reach depth
