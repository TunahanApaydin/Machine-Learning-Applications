import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

class Kmeans:

    def __init__(self):
        self.data = ""
    
    def k_means(self):
        self.data = pd.read_csv("customer.csv")
        X = self.data.iloc[:,3:].values
        #print(X)

        # En iyi K değerini bulmak için
        # results = []
        # for i in range(1, 11):
        #     kmeans = KMeans(n_clusters = i , init = "k-means++", random_state = 123)
        #     kmeans.fit(X)
        #     results.append(kmeans.inertia_)
        # plt.plot(range(1, 11), results)
        # plt.show()

        kmeans = KMeans(n_clusters = 4 , init = "k-means++", random_state = 123)
        y_pred = kmeans.fit_predict(X)

        plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 30, c = "blue")
        plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 30, c = "red")
        plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 30, c = "green")
        plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 30, c = "purple")
        plt.title("K-Means")
        plt.show()

if __name__ == "__main__":
    KM = Kmeans()
    KM.k_means()