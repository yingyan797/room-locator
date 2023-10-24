import numpy as np
from model import ValueInfo, OptimumFinder

# read dataset to np_array
cleanData = np.loadtxt("wifi_db/clean_dataset.txt")
noisyData = np.loadtxt("wifi_db/noisy_dataset.txt")
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
            
    def isLeaf(self):
        if self.right == None and self.left == None:
            return True
        else:
            return False


# count the total number of classes(labels) in the dataset
def count_labels(allData):
    labels = []
    for row in allData:
        l = row[-1]
        if l not in labels:
            labels.append(l)
    return len(labels)


label_count = count_labels(cleanData)


def sortColumn(dataset, attr):
    # sort the entire dataset on one attribute
    sortedDataset = dataset[dataset[:, attr].argsort()]
    return sortedDataset


def find_optimal_split_point(dataset):
    min_remainder = None  # This is the Information gained
    max_IG_attr = -1  # This is the attribute where the max IG value belongs
    max_IG_split = -1  # This is the split value where the max IG value appeared at
    for column in range(attr_count):
        sorted_dataset = sortColumn(dataset, column)  # sort the dataset
        label_frequency = [0 for i in range(label_count)]

        value = None
        value_group = []
        split_points = []
        for row in sorted_dataset:
            label = int(row[-1])
            v = row[column]
            label_frequency[label - 1] += 1  # calculate the frequency of this label
            if v == value:
                info = value_group[-1][1]
                info.frequency[label-1] += 1
                info.count += 1
            else:
                info = ValueInfo(label_count)
                info.frequency[label-1] = 1
                info.count = 1
                if len(value_group) >= 2:
                    info2 = value_group[-2][1]
                    info1 = value_group[-1][1]
                    if info2.unique_label() < 0 or info2.unique_label() != info1.unique_label():
                        split_points.append(v)
                value_group.append((v, info))
                value = v
        if len(value_group) >= 2:
            info2 = value_group[-2][1]
            info1 = value_group[-1][1]
            if info2.unique_label() < 0 or info2.unique_label() != info1.unique_label():
                split_points.append(v)

        left_finder = OptimumFinder(0, [0 for i in range(label_count)])
        right_finder = OptimumFinder(len(sorted_dataset), label_frequency)
        if min_remainder is None:
            min_remainder = right_finder.entropy()
        
        i = 0
        while len(split_points) > 0:
            (value, info) = value_group[i]
            if value < split_points[0]:
                left_finder.update(info, 1)
                right_finder.update(info, -1)
                i += 1
            else:
                split = split_points.pop(0)
                rem = (left_finder.entropy()*left_finder.size + right_finder.entropy()*right_finder.size) / len(sorted_dataset)
                # print(column, split, rem)
                if rem < min_remainder:
                    min_remainder = rem
                    max_IG_attr = column
                    max_IG_split = split
    return max_IG_attr, max_IG_split

def split_dataset(dataset, attr, split):#Get the left and right datasets from a splitting point
        left_set=dataset[dataset[:,attr] < split]
        right_set=dataset[dataset[:,attr] >= split]
        return np.array(left_set), np.array(right_set)

def decision_tree_learning(dataset, depth):
    all_labels = dataset[:, -1]  # Get the column with all the labels
    if np.all(all_labels == all_labels[0]) or depth >= 100:  # Checks if all of the labels are the same
        return Node(None, all_labels[0], None, None), depth
    else:
        attr, split = find_optimal_split_point(dataset)
        split_node = Node(attr, split, None, None)
        ldata, rdata = split_dataset(dataset, attr, split)
        split_node.left, ldepth = decision_tree_learning(ldata, depth+1)
        split_node.right, rdepth = decision_tree_learning(rdata, depth+1)
        return split_node, max(ldepth, rdepth)


def fit(dataset, depth):
    decision_tree = decision_tree_learning(dataset, depth)
    # TODO: Visualize tree
    return decision_tree

def predict(tree, dataset):
    # TODO:
    # Return array of class labels predicted using the input decision tree
    # Maintains same order
    labels = np.zeros(len(dataset))
    for i in range(len(dataset)):
        node = tree
        row = dataset[i]
        while not node.isLeaf():
            if row[node.attr] < node.val:
                node = node.left
            else:
                node = node.right
        labels[i] = node.val
    return labels
            

def cross_validation(dataset):
    #TODO:
    # Suffle dataset to maintain randomness
    np.random.shuffle(dataset)
    # Split dataset into 10 folds
    folds = np.array_split(dataset, 10)
    total = 0
    # Iterate 10 times, and each time do:
    for i in range(len(folds)):
        # Take 1 fold out as testing set
        testing_set = folds[i]
        remaining_folds = folds[:i] + folds[i+1:]
        training_set = remaining_folds[0]
        # Build decision tree based on training set (remaining 9 folds)
        tree, depth = decision_tree_learning(training_set, 0)
        # Store evaluation metrics for each iteration
        actual_class_labels = testing_set[:,-1]
        predicted_class_labels = predict(tree, testing_set)
        matrix = confusion_matrix(predicted_class_labels, actual_class_labels)
        acc = accuracy(matrix, len(predicted_class_labels))
        total += acc
    # Return averaged evalution metrics
    return total / 10

def confusion_matrix(predicted, actual):
    matrix = np.zeros((label_count, label_count))
    for i in range(len(actual)):
        pl = int(predicted[i] - 1)
        al = int(actual[i] - 1)
        matrix[al][pl] += 1
    return matrix

def accuracy(matrix, total):
    diagonal = 0
    for i in range(label_count):
        diagonal += matrix[i][i]
    return diagonal / total

def recall(matrix, total):
    return
    
# print(find_optimal_split_point(cleanData))
#tree, depth = decision_tree_learning(cleanData, 0)
#print(tree.show(), "depth", depth)
print(cross_validation(noisyData))