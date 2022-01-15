import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class PL:

    def __init__(self):
        self.data = ""
    
    def data_prep(self):
        self.data = pd.read_csv("salary.csv")
        # slice işlemleri
        edu_level = self.data.iloc[:,1:2] # dataframe
        salary = self.data.iloc[:,2:]
        
        e_values = edu_level.values # numpy array formatına çevirdik.
        s_values = salary.values
        print(s_values)

        #plt.scatter(e, s, color = "red") # dataset visualization.
        #plt.show()

        return e_values, s_values

    def polynomial_regression(self,  e_values, s_values, degree=4):
        poly_reg = PolynomialFeatures(degree = degree)
        e_poly = poly_reg.fit_transform(e_values) #verileri polinomal verilere dönüştürme.
        lin_reg = LinearRegression()
        lin_reg.fit(e_poly, s_values) #polinomal hale gelmiş veriler ile lineer regresyon.
        predict = lin_reg.predict(e_poly) #polinomal verilere göre eğitim.
        #print(predict)

        r2score = r2_score(s_values, lin_reg.predict(e_poly))
        print(r2score)

        #plt.scatter(e_values, s_values, color = "red") # veri setinin görselleştirilmesi.
        #plt.plot(e_values, predict, color = "blue") # modelin görselleştirilmesi.
        #plt.show()

        return predict

    def main(self):
        e_values, s_values = self.data_prep()
        self.polynomial_regression(e_values, s_values, degree=4)

if __name__ == "__main__":
    pl = PL()
    pl.main()
