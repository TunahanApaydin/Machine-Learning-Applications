import threading
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, accuracy_score, classification_report, plot_confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class IrisClassification:
    def __init__(self):
        self.data = ""
        self.x_train = ""
        self.x_test = ""
        self.y_train = ""
        self.y_test = ""
        self.t_size = 0.3
    
    def data_prep(self):
        self.data = pd.read_csv("iris.csv")
        #print(self.data.describe()) # numerical summary of each attribute

        indep_values = self.data.iloc[:,1:5].values
        dep_values = self.data.iloc[:,-1].values

        # lb_dep_values = label_binarize(dep_values, classes=[0, 1, 2])
        # n_classes = lb_dep_values.shape[1]

        return indep_values, dep_values

    def data_split(self, indep_values, dep_values):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(indep_values, dep_values, test_size = self.t_size, random_state = 0)
        #print(self.y_test)

    def logistic_reg(self):
        classifier = LogisticRegression()
        classifier.fit(self.x_train, self.y_train)
        lr_pred =  classifier.predict(self.x_test)
        #print(lr_pred)
        #acc = accuracy_score(self.y_test, lr_pred)
  
        lr_conf = confusion_matrix(self.y_test, lr_pred)
        
        proba = classifier.predict_proba(self.x_test)
        fpr, tpr, _ = roc_curve(self.y_test, proba[:,2], pos_label = "Iris-virginica")
        auc = roc_auc_score(self.y_test, proba, multi_class='ovr')

        plt.figure("confusion_matrix")
        plot_confusion_matrix(classifier, self.x_test, self.y_test, cmap=plt.cm.Blues)    
        plt.show()

        return fpr, tpr, auc
    
    def knn_reg(self):
        classifier = MultinomialNB()
        classifier.fit(self.x_train, self.y_train)
        knn_pred =  classifier.predict(self.x_test)
        #print(lr_pred)
        #acc = accuracy_score(self.y_test, lr_pred)
  
        knn_conf = confusion_matrix(self.y_test, knn_pred)
        
        proba = classifier.predict_proba(self.x_test)
        fpr, tpr, _ = roc_curve(self.y_test, proba[:,2], pos_label = "Iris-virginica")
        auc = roc_auc_score(self.y_test, proba, multi_class='ovr')

        # plt.figure("confusion_matrix")
        # metrics.plot_confusion_matrix(classifier, self.x_test, self.y_test, cmap=plt.cm.Blues)    
        # plt.show()

        return fpr, tpr, auc

    def visualize_ROC(self, roc_arr, names):
        plots = []
        countt=0
        plt.figure("Receiver Operating Characteristic - ROC")
        for i in range(3):
            for y in range(3):
                if countt==8:
                    break
                plots.append(plt.subplot2grid((3, 3), (i,y)))              
                plots[countt].plot(roc_arr[countt][0], roc_arr[countt][1], label = "area = {:.4f}".format(roc_arr[countt][2]))
                plots[countt].set(xlabel = "Specificity(False Positive Rate)", ylabel = "Sensitivity(True Positive Rate)")
                plots[countt].set_title(names[countt])
                countt+=1
        plt.tight_layout()
        plt.show()

    def visualize_Accuracy(self, names, accs):
        log_cols = ["Classifier", "Accuracy", "Log Loss"]
        log = pd.DataFrame(columns = log_cols)

        for i in range(8):
            log_entry = pd.DataFrame([[names[i], accs[i]*100, 11]], columns=log_cols)
            log = log.append(log_entry)

        sns.set_color_codes("muted")
        sns.barplot(x = "Accuracy", y = "Classifier", data=log, color = "b")
        plt.xlabel("Accuracy %")
        plt.title("Classifier Accuracy")
        plt.show()

    def visualize_Conf_Mat(self, classifier, names):
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,20))
        countt=0
        for clf, ax in zip(classifier, axes.flatten()):
            if countt == 8:
                break
            plot_confusion_matrix(clf, 
                          self.x_test, 
                          self.y_test, 
                          ax=ax, 
                          cmap='Blues',
                          colorbar = False)
            ax.title.set_text(names[countt])
            countt +=1
        plt.tight_layout()  
        plt.show()

    def model_comparison(self):
        classifier = [LogisticRegression(),
                    KNeighborsClassifier(n_neighbors = 5, metric = "minkowski"),
                    SVC(kernel = "rbf", probability = True),
                    GaussianNB(),
                    MultinomialNB(),
                    ComplementNB(),
                    DecisionTreeClassifier(criterion = "gini"),
                    RandomForestClassifier(n_estimators = 100, criterion = "gini")]

        accs = []
        names = []
        roc_arr = []
        for clf in classifier:
            clf.fit(self.x_train, self.y_train)

            name = clf.__class__.__name__

            # print("="*30)
            # print(name)
            # print("****Results****")

            predictions = clf.predict(self.x_test)
            acc = accuracy_score(self.y_test, predictions)
        
            #print("Accuracy: {:.4%}".format(acc))

            proba = clf.predict_proba(self.x_test)
            fpr, tpr, _ = roc_curve(self.y_test, proba[:,1], pos_label = "Iris-versicolor")
            auc = roc_auc_score(self.y_test, proba, multi_class = "ovr")

            arr=[]
            arr.append(fpr)
            arr.append(tpr)
            arr.append(auc)
            roc_arr.append(arr)

            names.append(name)
            accs.append(acc)

        return roc_arr, names, accs, classifier

    def main(self):
        indep_values, dep_values = self.data_prep()
        self.data_split(indep_values, dep_values)
        roc_arr, names, accs, classifier = self.model_comparison()
        #threading.Thread(self.visualize_ROC(roc_arr, names)).start()
        self.visualize_Accuracy(names, accs)
        self.visualize_Conf_Mat(classifier, names)

if __name__ == "__main__":
    iris_clf = IrisClassification()
    iris_clf.main()