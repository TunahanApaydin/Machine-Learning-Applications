import pandas as pd
import numpy as np
from MultipleLinearRegression import MLR
from sklearn import preprocessing

class mlrExample:

    def __init__(self):
        self.data = pd.read_csv("tennis.csv")
        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""

    def data_prep(self):
        #print(self.data.corr()) # parametrelerin birbiri üzerine olan etkisini gösteren korelasyon amtrisi üretir.
        outlook = self.data.iloc[:,0:1].values
        #windy = self.data.iloc[:,-2:-1].values
        #play = self.data.iloc[:,-1:].values
        #temperature = self.data.iloc[:,1:2].values
        #humadity = self.data.iloc[:,2:3].values
        temp_hum = self.data.iloc[:,1:3]
        #print(temp_hum)

        encoded_data = self.data.apply(preprocessing.LabelEncoder().fit_transform) #tüm kolonlara label encoder uygular.
        win_pl = encoded_data.iloc[:,-2:]

        le = preprocessing.LabelEncoder()
        ohe = preprocessing.OneHotEncoder()

        outlook[:,0] = le.fit_transform(self.data.iloc[:,0])
        outlook = ohe.fit_transform(outlook).toarray()

        weather = pd.DataFrame(data = outlook, index = range(14), columns = ["o","r","s"])
        df_data = pd.concat([win_pl, weather], axis=1)
        last_data = pd.concat([df_data, temp_hum], axis=1)
        print(last_data)

        return last_data

    def main(self):
        last_data = self.data_prep()
        mlr = MLR()
        r = mlr.OLS_result(last_data.iloc[:,:-1], last_data.iloc[:,-1:])
        new_data = mlr.backward_elimination(last_data, r)
        self.x_train, self.x_test, self.y_train, self.y_test = mlr.data_split(new_data.iloc[:,:-1], new_data.iloc[:,-1:])
        y_pred = mlr.multiple_linear_model(self.x_train , self.y_train, self.x_test, self.y_test)

if __name__ == "__main__":
    mlr = mlrExample()
    mlr.main()
