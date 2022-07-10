import numpy as np
import matplotlib.pyplot as plt

def trees_plot(error_list):
    index = np.arange(0,9)
    no_trees = [50, 100, 200, 300, 500, 1000, 2000, 5000, 10000]
    plt.plot(index, error_list, label = "accuracy", linestyle="--")
    plt.xlabel('No. of Trees')
    plt.ylabel('OOB_Error')
    plt.xticks(index, no_trees)
    plt.title('Performance of Random Forest on different number of trees')
    plt.legend()
    plt.savefig('/home/z5209394/ew-honoursthesis/Graphs/no_trees.png')
