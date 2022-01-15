import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

class HierarchicalClustering:

    def __init__(self):
        self.data = ""
    
    def agglomerative_clustering(self):
        self.data = pd.read_csv("customer.csv")
        X = self.data.iloc[:,3:].values
        #print(X)

        agglomerative = AgglomerativeClustering(n_clusters = 3, affinity = "euclidean", linkage = "ward")
        y_pred = agglomerative.fit_predict(X)
        print(y_pred)

        plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 40, c = "blue")
        plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 40, c = "red")
        plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 40, c = "green")
        plt.show()

        dendrogram = hierarchy.dendrogram(hierarchy.linkage(X, method = "ward"))
        plt.show()

        

if __name__ == "__main__":
    HC = HierarchicalClustering()
    HC.agglomerative_clustering()
    