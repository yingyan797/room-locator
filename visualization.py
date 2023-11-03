import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

class Wifi_Visualizer:
    def __init__(self, decision):
        self.attr = decision.attr_count
        self.data_name = decision.data_name[8:-4]
        self.all_data = decision.all_data
        self.ref_vector = np.array([1]+[0 for i in range(self.attr-1)])
        self.colors = ['b', 'g', 'r', 'm']
    
    def polar_coord(self, row): # transform each row of dataset to 2D polar coordinate
        d = 0
        for x in row[:-1]:
            d += x*x
        dist = np.sqrt(d)
        ang = np.dot(self.ref_vector, row[:-1])/dist
        return dist, ang, row[-1]
    
    def plot_dataset(self):
        ds = [[] for i in range(4)]
        angs = [[] for i in range(4)]
        for row in self.all_data:
            d,a,l = self.polar_coord(row)
            ds[int(l)-1].append(d)
            angs[int(l)-1].append(a)
        fn = "static/wifi_visual/"+self.data_name
        plt.title(self.data_name+" in polar coordinate")
        plt.xlabel("Vector norm")
        plt.ylabel("Polar angle cosine")
        for i in range(4):
            plt.scatter(ds[i], angs[i], c=self.colors[i])
        plt.savefig(fn)
        return fn+".png"

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
        graphs = []
        f = open("graphdb.csv", "a")    # The database for decision tree graph names
        # Congigure plots
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
