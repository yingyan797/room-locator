import numpy as np
from decision_tree import Decision, cleanData, noisyData
from visualization import Tree_Visualizer


if __name__ == '__main__':
    dt1 = Decision()
    dt2 = Decision()
    # read dataset to np_array
    print("Loading the data:.......\n")
    dt1.load_data(cleanData)
    dt2.load_data(noisyData)

    print("Creating the decision tree according to the data.....\n")
    # print(find_optimal_split_point(cleanData))
    dt1.fit()
    dt2.fit()
    # print(tree.show(), "depth", depth)
    print("Cross validating:.......... \n")
    print("Clean Data:\n")
    mt, matrix, acc, table = dt1.cross_validation()
    print(matrix, "\nAccuracy:", acc, "\nPrf metrics",table)
    print("Noisy Data:\n")
    mt, matrix, acc, table = dt2.cross_validation()
    print(matrix, "\nAccuracy:", acc, "\nPrf metrics",table)
