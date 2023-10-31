import matplotlib.pyplot as plt
import numpy as np

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
                self.subtree_list.append(node)
                plt.text(node.x, node.y, f"subT{len(self.subtree_list)}",
                         bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "w", "edgecolor": "k"}, fontsize="large")
            else:
                if node.attr is not None:
                    plt.text(node.x, node.y, f"WiFi {node.attr}\n <{node.val}",
                             bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "w", "edgecolor": "k"},
                             fontsize="large")
                else:
                    plt.text(node.x, node.y, f" Room:\n {node.val}",
                             bbox={"boxstyle": "round", "pad": 0.4, "facecolor": "w", "edgecolor": "k"},
                             fontsize="large")
                self.plot_tree(node.left, level - 1, position - 2 ** level / 10, (node.x, node.y))
                self.plot_tree(node.right, level - 1, position + 2 ** level / 10, (node.x, node.y))

    def visualize(self, session_num, data_name):
        # Configure plot
        graphs = []
        f = open("graphdb.csv", "a")    
        # The database for recording the dataset from which each decision tree plot is generated
        fig, ax = plt.subplots(figsize=(30, 30))
        plt.axis('off')
        ax.set_aspect('equal')
        self.plot_tree(self.tree, level=5, position=0)
        gname = "mainT_"+str(session_num)+".png"
        plt.savefig("static/plots/"+gname)
        f.write(str(session_num)+','+data_name+','+gname+'\n')
        graphs.append(gname)

        for index, t in enumerate(self.subtree_list):
            fig, ax = plt.subplots(figsize=(30, 30))
            plt.axis('off')
            ax.set_aspect('equal')
            self.plot_tree(t, level=5, position=0)
            gname = "subT"+str(index+1)+'_'+str(session_num)+".png"
            plt.savefig("static/plots/"+gname)
            f.write(str(session_num)+','+data_name+','+gname+'\n')
            graphs.append(gname)
        f.close()
        return graphs

# if __name__ == '__main__':
#     cleanData = np.loadtxt("wifi_db/clean_dataset.txt")
#     tree, depth = decision_tree_learning(cleanData, 0, 10)
#     visualizer = Tree_Visualizer(tree)
#     visualizer.visualize()
