import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

data=pd.read_csv("spam.csv",encoding="latin-1")
data=data[['v1','v2']]
data.columns=['label','msg']# changing headed names

data['label']=data['label'].map({'ham':0,'spam':1})

x_train,x_test,y_train,y_test=train_test_split(data['msg'],data['label'],test_size=0.2)

vectorizer=CountVectorizer()
x_train=vectorizer.fit_transform(x_train)
x_test=vectorizer.transform(x_test)

#Naive Bayes
model=MultinomialNB()
model.fit(x_train,y_train)

prd=model.predict(x_test)
print("Accuracy(Naive Bayes):",accuracy_score(y_test,prd))
print(classification_report(y_test,prd))

#Decision Tree
model2=DecisionTreeClassifier()
model2.fit(x_train,y_train)

prd2=model.predict(x_test)
print("Accuracy (Decision Tree):",accuracy_score(y_test,prd2))
print(classification_report(y_test,prd2))
