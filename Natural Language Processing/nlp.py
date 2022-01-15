import re
import nltk
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer

class nlpExample:

    def __init__(self):
        self.data = ""
        self.X_train = ""
        self.X_test = ""
        self.y_train = ""
        self.y_test = ""

    def data_prep(self):
        self.data = pd.read_csv("Tweet_Dataset.csv")
        #print(self.data["Tweets"])
        #nltk.download("stopwords")

        # with open("Tweet_Dataset.csv", "r") as file:
        #     comment = []
        #     score = []

        #     for line in file:
        #         line = line.rstrip(",")
        #         print(line[:-1])
        #         comment.append(line[:-1])
        #         score.append(line[-1])
        # comment = pd.DataFrame(comment,columns=["Tweets"])
        # score = pd.DataFrame(score,columns=["Gender"])    
        # self.data = pd.concat([comment,score],axis=1)
        # print(score)
        ps = PorterStemmer()
        print(set(stopwords.words("turkish")))
        processed_rewiev = []
        for i in range(0, 115, 1):
            rewiev = re.sub("[^a-zA-Z]", " ", self.data["Tweets"][i]) # noktalama işaretlerini siler.
            rewiev = rewiev.lower() # Tüm kelimeleri küçük harf yapar.
            rewiev =  rewiev.split() # Cümleleri liste haline çevirir.
             
            rewiev = [ps.stem(word) for word in rewiev if not word in set(stopwords.words("turkish"))]
            rewiev = " ".join(rewiev)

            processed_rewiev.append(rewiev)
        print(rewiev)

        return processed_rewiev
    
    def feature_extraction(self, processed_rewiev): # Bag of Words (BOW)
        cv = CountVectorizer(max_features = 50)

        X = cv.fit_transform(processed_rewiev).toarray() # count vector
        y = self.data.iloc[:,1].values

        return X, y

    def data_split(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    def GaussianNB_Classifier(self):
        gnb = GaussianNB()

        gnb.fit(self.X_train, self. y_train)
        y_pred = gnb.predict(self.X_test)

        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)

if __name__ == "__main__":
    nlp = nlpExample()
    processed_rewiev = nlp.data_prep()
    X, y = nlp.feature_extraction(processed_rewiev)
    nlp.data_split(X, y)
    nlp.GaussianNB_Classifier()