import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

class RFR:
    def __init__(self):
        self.data = ""
    
    def data_prep(self):
        self.data = pd.read_csv("salary.csv")
        # slice işlemleri
        edu_level = self.data.iloc[:,1:2] # dataframe
        salary = self.data.iloc[:,2:]
        
        e_values = edu_level.values # numpy array formatına çevirdik.
        s_values = salary.values

        return e_values, s_values
    
    def random_forest_regression(self, e_values, s_values):
        rf_reg = RandomForestRegressor(n_estimators = 10, random_state  =0) # n_estimators: kaç tane karar ağacı oluşturulacağını belirler.
        rf_reg.fit(e_values, s_values.ravel())
        print(rf_reg.predict([[6.6]]))
        predict = rf_reg.predict(e_values)

        r2score = r2_score(s_values, rf_reg.predict(e_values)) # r_square hesaplama
        print("s:", r2score)

        #plt.scatter(e_values, s_values, color="red")
        #plt.plot(e_values , predict, color="blue")
        #plt.show()

    def main(self):
        e_values, s_values = self.data_prep()
        self.random_forest_regression(e_values, s_values)


if __name__ == "__main__":
    rfr = RFR()
    rfr.main()

