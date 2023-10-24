import matplotlib.pyplot as plt
import numpy as np
from decision_tree import decision_tree_learning


class Tree_Visualizer:

    def __init__(self, my_tree):
        self.tree = my_tree
        self.subtree_list = []

    def plot_tree(self, node, level=0, position=0, parent_coords=None):
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        if node is not None:
            node.x = position
            node.y = level
            if parent_coords:
                plt.plot([parent_coords[0], node.x], [parent_coords[1], node.y], '-',
                         color=color[level % len(color)])
            if level < 1 and (node.attr is not None):
                print (node.attr)
                self.subtree_list.append(node)
                plt.text(node.x, node.y, f"subT{len(self.subtree_list)}", ha='center',
                         bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "w", "edgecolor": "k"}, fontsize="small")
            else:
                if node.attr is not None:
                    plt.text(node.x, node.y, f"attr: {node.attr}\n <{node.val}", ha='center',
                             bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "w", "edgecolor": "k"},
                             fontsize="small")
                else:
                    plt.text(node.x, node.y, f" Room:\n {node.val}", ha='center',
                             bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "w", "edgecolor": "k"},
                             fontsize="small")
                self.plot_tree(node.left, level - 1, position - 2 ** level / 10, (node.x, node.y))
                self.plot_tree(node.right, level - 1, position + 2 ** level / 10, (node.x, node.y))

    def visualize(self):
        # Configure plot
        fig, ax = plt.subplots(figsize=(30, 30))
        plt.axis('off')
        ax.set_aspect('equal')
        self.plot_tree(self.tree, level=5, position=0)
        plt.savefig("./plots/mainT.png")
        for index, t in enumerate(self.subtree_list):
            fig, ax = plt.subplots(figsize=(30, 30))
            plt.axis('off')
            ax.set_aspect('equal')
            self.plot_tree(t, level=5, position=0)
            plt.savefig(f"./plots/subT{index+1}.png")


if __name__ == '__main__':
    cleanData = np.loadtxt("wifi_db/clean_dataset.txt")
    tree, depth = decision_tree_learning(cleanData, 0)
    visualizer = Tree_Visualizer(tree)
    visualizer.visualize()
