import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score

class SVM:
    
    def __init__(self):
        self.data = ""
    
    def data_prep(self):
        self.data = pd.read_csv("salary.csv")
        # slice işlemleri
        edu_level = self.data.iloc[:,1:2] # dataframe
        salary = self.data.iloc[:,2:]
        
        e_values = edu_level.values # numpy array formatına çevirdik.
        s_values = salary.values

        ss1 = StandardScaler()
        e_val_scaled = ss1.fit_transform(e_values)
        ss2 = StandardScaler()
        s_val_scaled = np.ravel(ss2.fit_transform(s_values.reshape(-1,1)))
        #s_val_scaled = ss2.fit_transform(s_values)

        #plt.scatter(e_values, s_values, color = "red") # dataset visualization.
        #plt.show()

        return e_values, s_values

    def support_vector_regression(self, e_values, s_values):
        ss1 = StandardScaler()
        e_val_scaled = ss1.fit_transform(e_values)
        ss2 = StandardScaler()
        s_val_scaled = np.ravel(ss2.fit_transform(s_values.reshape(-1,1)))
        
        svr_reg = SVR(kernel="rbf")
        svr_reg.fit(e_val_scaled, s_val_scaled)
        predict = ss2.inverse_transform(svr_reg.predict(ss1.transform(np.array([[4]]))))
        print(predict)

        r2score = r2_score(s_val_scaled, svr_reg.predict(e_val_scaled))
        print(r2score)

        #plt.scatter(e_val_scaled, s_val_scaled, color="red")
        #plt.plot(e_val_scaled, predict, color="blue")
        #plt.show()

    def main(self):
        e_values, s_values = self.data_prep()
        self.support_vector_regression(e_values, s_values)

if __name__ == "__main__":
    svr = SVM()
    svr.main()