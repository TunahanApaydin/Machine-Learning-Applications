import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class DTR:

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
    
    def decision_tree_regression(self, e_values, s_values):
        dt_reg = DecisionTreeRegressor(random_state=0)
        dt_reg.fit(e_values, s_values)
        predict = dt_reg.predict(e_values)
        print(dt_reg.predict([[11]]))
        print(dt_reg.predict([[6.6]]))

        r2score = r2_score(s_values, dt_reg.predict(e_values))
        print(r2score)

        #plt.scatter(e_values, s_values, color="red")
        #plt.plot(e_values, predict, color="blue")
        #plt.show()

    def main(self):
        e_values, s_values = self.data_prep()
        self.decision_tree_regression(e_values, s_values)

if __name__ == "__main__":
    dtr = DTR()
    dtr.main()