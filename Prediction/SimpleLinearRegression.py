import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class SLR:

    def __init__(self):
        self.data = pd.read_csv('sales.csv')
        self.X_train = ""
        self.X_test = ""
        self.Y_train = ""
        self.Y_test = ""
        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""

    def data_preprocessing(self):
        months = self.data[['Aylar']]
        sales = self.data[['Satislar']]

        return months, sales

    def data_split(self, months, sales):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(months, sales, test_size=0.33, random_state=0)
        # ss = StandardScaler()
        # self.X_train = ss.fit_transform(x_train)
        # self.X_test = ss.fit_transform(x_test)
        # self.Y_train = ss.fit_transform(y_train)
        # self.Y_test = ss.fit_transform(y_test)
        print("Y_test",self.y_test)
        print("-------------------------------------")

    def linear_model(self):
        lr = LinearRegression()
        lr.fit(self.x_train, self.y_train)

        prediction = lr.predict(self.x_test)
        print("Prediction",prediction)

        return prediction
    
    def visualization(self, prediction):
        x_train = self.x_train.sort_index()
        y_train = self.y_train.sort_index()
        plt.title("Sales Amount by Month")
        plt.xlabel("Months")
        plt.ylabel("Sales")
        plt.plot(x_train, y_train)
        plt.plot(self.x_test, prediction)
        plt.show()

    def main(self):
        months , sales = self.data_preprocessing()
        self.data_split(months, sales)
        prediction = self.linear_model()
        self.visualization(prediction)

if __name__ == "__main__":
    slr = SLR()
    slr.main()