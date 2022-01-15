import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class MLR:

    def __init__(self):
        self.data = ""
        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""

    def data_preprocessing(self):
        self.data = pd.read_csv("veriler.csv")
        #cinsiyet = self.data[["cinsiyet"]]
        cinsiyet = self.data.iloc[:,-1:].values
        ulke = self.data.iloc[:,0:1].values
        bky = self.data.iloc[:,1:4]

        le = preprocessing.LabelEncoder()
        ohe = preprocessing.OneHotEncoder()

        cinsiyet[:,-1] =  le.fit_transform(self.data.iloc[:,-1])
        cinsiyet = ohe.fit_transform(cinsiyet).toarray()
        ulke[:,0] = le.fit_transform(self.data.iloc[:,0])
        ulke = ohe.fit_transform(ulke).toarray()
        
        sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ["fr", "tr", "us"])
        sonuc2 = pd.DataFrame(data = bky, index = range(22), columns = ["boy", "kilo", "yas"])
        sonuc3 = pd.DataFrame(data = cinsiyet[:,:1], index = range(22), columns = ["c"])
        
        s = pd.concat([sonuc, sonuc2], axis = 1)
        s2 = pd.concat([s, sonuc3], axis = 1)
        
        # boy = s2.iloc[:,3:4].values
        # sol = s2.iloc[:,:3].values #values alırsan değerleri çektiğin için aşağıdaki gibi tekrar DataFrame e dönüştürmen gerekir.
        # sag  = s2.iloc[:,4:].values #direk satır sütünları çekersen(.iloc) zaten DataFrame olarak almış olursun.

        # boy_df = pd.DataFrame(data = boy, index=range(22), columns=["boy"])
        # sol_df = pd.DataFrame(data = sol, index=range(22), columns=["fr", "tr", "us"])
        # sag_df = pd.DataFrame(data = sag, index=range(22), columns=["kilo", "yas", "c"])

        boy = s2.iloc[:,3:4]
        sol = s2.iloc[:,:3]
        sag  = s2.iloc[:,4:]

        veri = pd.concat([sol, sag], axis=1) # axis = 1 sütün olarak birleştirir.
        print(veri)

        return veri, boy
    
    def data_split(self, veri, dep_val):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(veri, dep_val, test_size=0.33, random_state=0)
        print(self.y_test)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def multiple_linear_model(self, x_train , y_train, x_test, y_test):
        self.x_train, self.y_train, self.x_test, self.y_test = x_train , y_train, x_test, y_test
        regressor = LinearRegression()
        #regressor2 = LinearRegression()

        #regressor.fit(self.x_train, self.y_train)
        #y_pred = regressor.predict(self.x_test)

        regressor.fit(self.x_train, self.y_train)
        y_pred = regressor.predict(self.x_test)
        print(y_pred)

        r2score = r2_score(self.y_test, y_pred)
        print(r2score)

        return y_pred

    def OLS_result(self, veri, dep_val):
        #array = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis = 1)
        array_list = veri.iloc[:,[0,1,2,3,4,5]].values
        r_ols = sm.OLS(endog = dep_val, exog = array_list)
        r = r_ols.fit()
        print(r.summary())
        return r

    def backward_elimination(self, veri, r):
        p_values = list(r.pvalues)
        max_p = max(p_values)
        max_p_idx = p_values.index(max_p)

        if len(veri.columns)-1 == max_p_idx:
            veri = veri.iloc[:,:max_p_idx]
            print(veri)
        elif max_p_idx == 0:
            veri = veri.iloc[:,(max_p_idx + 1):]
            print(veri)
        else:
            array_sol = veri.iloc[:,:max_p_idx]      
            array_right = veri.iloc[:,(max_p_idx + 1):]       
            veri = pd.concat([array_sol, array_right], axis=1)
            print(veri)

        return veri
    
    def backward_elimination2(self, veri, r):
        #p_values = r.summary2().tables[1]['P>|t|']
        attributeIndex=0
        while attributeIndex < len(veri.columns):
            p_values = r.pvalues[attributeIndex]
            if p_values > 0.05:
                if len(veri.columns)-1 == attributeIndex:
                    veri = veri.iloc[:,:attributeIndex]
                    attributeIndex=0
                    print(veri)
                    continue
                elif attributeIndex == 0:
                    veri = veri.iloc[:,(attributeIndex + 1):]
                    attributeIndex=0
                    print(veri)
                    continue
                else:
                    array_sol = veri.iloc[:,:attributeIndex]      
                    array_right = veri.iloc[:,(attributeIndex + 1):]       
                    veri = pd.concat([array_sol, array_right], axis=1)
                    attributeIndex=0
                    print(veri)
                    continue

            attributeIndex=attributeIndex+1
        return veri

    def main(self):
        veri, boy = self.data_preprocessing()
        veri = self.OLS_result(veri, boy)
        self.data_split(veri, boy)
        self.multiple_linear_model()

if __name__ == "__main__":
    mlr = MLR()
    mlr.main()