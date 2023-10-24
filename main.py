import numpy as np
from decision_tree import decision_tree_learning, cross_validation
from visualization import Tree_Visualizer


if __name__ == '__main__':
    # read dataset to np_array
    print("Loading the data:.......\n")
    cleanData = np.loadtxt("wifi_db/clean_dataset.txt")
    noisyData = np.loadtxt("wifi_db/noisy_dataset.txt")
    data_shape = cleanData.shape
    attr_count = data_shape[1] - 1

    print("Creating the decision tree according to the data.....\n")
    # print(find_optimal_split_point(cleanData))
    tree, depth = decision_tree_learning(cleanData, 0, 10)
    # print(tree.show(), "depth", depth)
    print("Cross validating:.......... \n")
    print("Clean Data:\n")
    acc, maxtree = cross_validation(cleanData)
    print("Noisy Data:\n")
    cross_validation(noisyData)

    print("Visualizing tree and storing plot to plot..........\n")
    tree_visualizer = Tree_Visualizer(tree)
    tree_visualizer.visualize()

