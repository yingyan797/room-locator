import numpy as np
from model import ValueInfo, OptimumFinder

cleanData = "wifi_db/clean_dataset.txt"
noisyData = "wifi_db/noisy_dataset.txt"

# define the data structure of the decision tree
class Node:
    def __init__(self, attr, val, left, right):
        self.attr = attr    # attribute (WiFi) number to split on
        self.val = val      # WiFi signal value to split on
        self.left = left    # subtree with value < current node
        self.right = right  # subtree with value >= current node

    def show(self):         # Text representation of decision tree
        if self.left is None and self.right is None:
            return "Then the label is " + str(self.val)
        return "If attribute " + str(self.attr) + " <= " + str(
            self.val) + " {\n  " + self.left.show() + "\n}\n" + "otherwise (attribute " + str(self.attr) + ") > " + str(
            self.val) + " {\n  " + self.right.show()

    def isLeaf(self):       # it is leaf node if both left and right subtrees are None
        if self.right == None and self.left == None:
            return True
        else:
            return False

# Get the left and right datasets from a splitting point
def split_dataset(dataset, attr, split):  
    left_set = dataset[dataset[:, attr] < split]
    right_set = dataset[dataset[:, attr] >= split]
    return np.array(left_set), np.array(right_set)

# Count the total number of classes (labels) in the dataset
def count_labels(allData):
    labels = []
    for row in allData:
        l = row[-1]
        if l not in labels:
            labels.append(l)
    return len(labels)

# Sort the entire dataset on one attribute
def sortColumn(dataset, attr):
    sortedDataset = dataset[dataset[:, attr].argsort()]
    return sortedDataset

# Return an array of class labels predicted for input dataset using the selected decision tree
# Maintains same order
def predict(tree, dataset):
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

# Main class for processing training dataset, cross validation, and create decision trees
class Decision:
    def __init__(self):
        self.all_data = None
        self.decision_tree = None
        self.data_name = None   # File name of input dataset
        self.prune_depth = None
        self.prune_pct = None

    def load_data(self, data_file):     # Read .txt data file into array and count the number labels
        self.data_name = data_file
        self.all_data = np.loadtxt(data_file)
        self.label_count = count_labels(self.all_data)
        self.attr_count = len(self.all_data[0]) - 1

    def apply_pruning(self, depth, percentage):
        if not depth:
            self.prune_depth = 7
        else:
            self.prune_depth = int(depth)
        if not percentage:
            self.prune_pct = 95
        else:
            self.prune_pct = float(percentage)

    def find_optimal_split_point(self, dataset):
        min_remainder = None  # The splitting option yielding minimum remainder entropy, meaning the maximum information gain
        max_IG_attr = -1  # This is the attribute for splitting where the maximum information gain belongs
        max_IG_split = -1  # This is the split value where the maximum information gain appeared
        for column in range(self.attr_count):
            sorted_dataset = sortColumn(dataset, column)  # sort the dataset
            label_frequency = [0 for i in range(self.label_count)]

            value = None
            value_group = []        # For each distinct value in dataset, record its information in ValueInfo object
            split_points = []       # Find every value where splitting is considered
            for row in sorted_dataset:
                label = int(row[-1])
                v = row[column]
                label_frequency[label - 1] += 1  # increment the frequency of this label
                if v == value:      # The value is repeated
                    info = value_group[-1][1]
                    info.frequency[label - 1] += 1
                    info.count += 1
                else:               # The value is never encountered
                    info = ValueInfo(self.label_count)  # create ValueInfo object
                    info.frequency[label - 1] = 1
                    info.count = 1
                    if len(value_group) >= 2:
                        info2 = value_group[-2][1]
                        info1 = value_group[-1][1]
                        if info2.unique_label() < 0 or info2.unique_label() != info1.unique_label():
                            split_points.append(v)      # Criteria for whether a value should be considered splitting
                    value_group.append((v, info))
                    value = v
            if len(value_group) >= 2:                   # Append the last value
                info2 = value_group[-2][1]
                info1 = value_group[-1][1]
                if info2.unique_label() < 0 or info2.unique_label() != info1.unique_label():
                    split_points.append(v)

            # Using two OptimumFinder objects for left and right subtrees respectively
            left_finder = OptimumFinder(0, [0 for i in range(self.label_count)])
            right_finder = OptimumFinder(len(sorted_dataset), label_frequency)
            if min_remainder is None:
                min_remainder = right_finder.entropy()

            i = 0
            while len(split_points) > 0:        # Iteratively try each split value considered
                (value, info) = value_group[i]
                if value < split_points[0]:
                    left_finder.update(info, 1)     # Left subtree add the current value and its info
                    right_finder.update(info, -1)   # Right subtree deduct the current value and its info
                    i += 1
                else:                               # Split the dataset on this value
                    split = split_points.pop(0)
                    rem = (left_finder.entropy() * left_finder.size + right_finder.entropy() * right_finder.size) / len(
                        sorted_dataset)
                    if rem < min_remainder:         # find minimum remainder
                        min_remainder = rem
                        max_IG_attr = column
                        max_IG_split = split
        return max_IG_attr, max_IG_split

    def decision_tree_learning(self, dataset, depth, maxdepth):
        label_freq = [0 for i in range(self.label_count)]
        for label in dataset[:, -1]:
            label_freq[int(label-1)] += 1
        
        dominant_pct = 0
        dominant_label = 0
        for i in range(len(label_freq)):
            pct = label_freq[i]/len(dataset)
            if pct > dominant_pct:
                dominant_pct = pct
                dominant_label = i+1
              
        if depth > maxdepth or dominant_pct == 1 or (
            self.prune_depth and depth >= self.prune_depth and dominant_pct*100 >= self.prune_pct):
            return Node(None, dominant_label, None, None), depth

        else:       # recursively find optimum splitting points for left and right subtrees
            attr, split = self.find_optimal_split_point(dataset)
            split_node = Node(attr, split, None, None)
            ldata, rdata = split_dataset(dataset, attr, split)
            split_node.left, ldepth = self.decision_tree_learning(ldata, depth + 1, 10)
            split_node.right, rdepth = self.decision_tree_learning(rdata, depth + 1, 10)
            return split_node, max(ldepth, rdepth)


    def fit(self):  # create decision tree on the entire dataset
        self.decision_tree = self.decision_tree_learning(self.all_data, 0, 10)[0]

    def cross_validation(self):
        # Suffle dataset to maintain randomness
        nb_folds = 10
        np.random.shuffle(self.all_data)
        # Split dataset into 10 folds
        folds = np.array_split(self.all_data, 10)
        totalAccuracy = 0
        max = (-1, None)
        matrix = np.zeros((self.label_count, self.label_count))

        # Iterate 10 times, and each time do:
        for i in range(len(folds)):
            # Take 1 fold out as testing set
            testing_set = folds[i]
            remaining_folds = folds[:i] + folds[i + 1:]
            training_set = remaining_folds.pop(0)
            while len(remaining_folds) > 0:
                training_set = np.concatenate((training_set, remaining_folds.pop(0)))
            
            # Build decision tree based on training set (remaining 9 folds)
            tree, depth = self.decision_tree_learning(training_set, 0, 10)
            # Store evaluation metrics for each iteration
            actual_class_labels = testing_set[:, -1]
            predicted_class_labels = predict(tree, testing_set)
            fold_matrix = self.confusion_matrix(predicted_class_labels, actual_class_labels)
            matrix += fold_matrix   # adding the current frequencies to the total confusion matrix
            curr_acc = self.accuracy(fold_matrix, len(predicted_class_labels))
            totalAccuracy += curr_acc
            #record the most accurate tree and return the result and the tree
            if curr_acc >= max[0]:
                max = (curr_acc, tree)
                
        matrix /= nb_folds

        precisions, recalls, f1s = self.prf_metrics(matrix)
        prf_table = [["", "Precision", "Recall", "F1-measure"]]+[
            ["Room "+str(i+1), str(precisions[i] * 100) +"%", str(recalls[i] * 100) +"%", str(f1s[i] * 100) +"%" ]
            for i in range(self.label_count)]

        # Return the decision tree with highest accuracy, averaged evaluation metrics, and confusion matrix
        return max, matrix, str(totalAccuracy / nb_folds * 100)+"%", prf_table
 
    def confusion_matrix(self, predicted, actual):
        matrix = np.zeros((self.label_count, self.label_count))
        for i in range(len(actual)):
            pl = int(predicted[i] - 1)
            al = int(actual[i] - 1)
            matrix[al][pl] += 1
        return matrix


    def accuracy(self, matrix, total):
        diagonal = 0
        for i in range(self.label_count):
            diagonal += matrix[i][i]
        return diagonal / total


    def prf_metrics(self, matrix):      # calculate p(recesion), r(ecall), and f(1measure)
        recalls, precisions, f1s = [], [], []
        for i in range(self.label_count):
            tp = matrix[i][i]
            totalRow = np.sum(matrix[i])
            totalCol = np.sum(matrix[:, i])
            recall = tp / totalRow
            recalls.append(recall)
            precision = tp / totalCol
            precisions.append(precision) 
            f1s.append(2 * precision * recall / (precision + recall))
        return precisions, recalls, f1s

# dt = Decision()
# dt.load_data(noisyData)
# dt.cross_validation()
