import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

class SupVecClass:
    def __init__(self):
        self.data = ""
        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""
        self.t_size = 0.33
    
    def data_prep(self):
        self.data = pd.read_csv("veriler.csv")

        bky = self.data.iloc[:,1:4].values
        gender = self.data.iloc[:,4:].values
        
        #le = preprocessing.LabelEncoder()
        #labeled_gender = gender[:,-1] =  le.fit_transform(self.data.iloc[:,-1])

        return bky, gender

    def data_split(self, indep_val, dep_val, t_size):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(indep_val, dep_val, test_size = t_size, random_state = 0)
        print(self.y_test)

    def support_vector_classifier(self):
        ss = StandardScaler()
        scaled_x_train = ss.fit_transform(self.x_train) # öğrenip transform et
        scaled_x_test = ss.transform(self.x_test) # fit yok yani öğrenme sadece transform et

        svc = SVC(kernel = "rbf")
        svc.fit(scaled_x_train, self.y_train)
        predict = svc.predict(scaled_x_test)
        print(predict)

        conf_matrix = confusion_matrix(self.y_test, predict)
        print(conf_matrix)

    def main(self):
        indep_val, dep_val = self.data_prep()
        self.data_split(indep_val, dep_val, self.t_size)
        self.support_vector_classifier()

if __name__ == "__main__":
    svc = SupVecClass()
    svc.main()