import matplotlib.pyplot as plt
import numpy as np
from decision_tree import Node, decision_tree_learning


def plot_tree(node, level=0, position=0, parent_coords=None):
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if node is not None:
        node.x = position
        node.y = level
        if parent_coords:
            plt.plot([parent_coords[0], node.x], [parent_coords[1], node.y], '-',
                     color=color[level % len(color)])
        plt.text(node.x, node.y, f"{node.attr}\n{node.val}", ha='center',
                 bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "w", "edgecolor": "k"})
        plot_tree(node.left, level - 1, position - 2 ** level, (node.x, node.y))
        plot_tree(node.right, level - 1, position + 2 ** level, (node.x, node.y))


# Initialize tree nodes
root = Node('Root', 1)
root.left = Node('L1', 2)
root.right = Node('R1', 3)
root.left.left = Node('LL1', 4)
root.left.right = Node('LR1', 5)
root.right.right = Node('RR1', 6)

# Configure plot
fig, ax = plt.subplots()
plt.axis('off')
ax.set_aspect('equal')

# Plot tree
# plot_tree(root, level=0, position=0)

cleanData = np.loadtxt("wifi_db/clean_dataset.txt")

tree, depth = decision_tree_learning(cleanData, 0)
plot_tree(tree, level=0, position=0)
plt.show()
