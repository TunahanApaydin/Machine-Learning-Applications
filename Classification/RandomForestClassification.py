import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, RocCurveDisplay


class RFC:
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

    def random_forest_classifier(self):
        ss = StandardScaler()
        scaled_x_train = ss.fit_transform(self.x_train) # öğrenip transform et
        scaled_x_test = ss.transform(self.x_test) # fit yok yani öğrenme sadece transform et

        rfc = RandomForestClassifier(n_estimators = 5, criterion = "gini")
        rfc.fit(scaled_x_train, self.y_train)
        predict = rfc.predict(scaled_x_test)
        print(predict)
        proba = rfc.predict_proba(scaled_x_test)
        conf_mat = confusion_matrix(self.y_test, predict) # tn, fp, fn, tp = confusion_matrix(self.y_test, predict).ravel()
        print(conf_mat)

        fpr, tpr, threshold =  roc_curve(self.y_test, proba[:,0], pos_label = "e")
        print(fpr)
        print(tpr)

        auc = roc_auc_score(self.y_test, proba[:,0])
        print(auc)

        return fpr, tpr
        
    def visualize_ROC(self, fpr, tpr):
        fig, ax = plt.subplots(figsize=(10,7))
        ax.plot(fpr, tpr)
        ax.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), label='baseline', linestyle='--') # np.linspace(0, 1, 100) >> 0 -1 arasını 100 e böler.
        plt.title('Receiver Operating Characteristic Curve', fontsize=18)
        plt.ylabel('TPR', fontsize=16)
        plt.xlabel('FPR', fontsize=16)
        plt.legend(fontsize=12)
        plt.show()

    def main(self):
        indep_val, dep_val = self.data_prep()
        self.data_split(indep_val, dep_val, self.t_size)
        fpr, tpr = self.random_forest_classifier()
        self.visualize_ROC(fpr, tpr)

if __name__ == "__main__":
    dtc = RFC()
    dtc.main()