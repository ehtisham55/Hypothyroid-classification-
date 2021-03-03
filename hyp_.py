import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

df=pd.read_csv("hypothyroid.csv")

df["Age"].replace({"?": 50}, inplace=True)
df["Sex"].replace({"?": "M"}, inplace=True)
df["TSH"].replace({"?": 0}, inplace=True)
df["T3"].replace({"?": 0}, inplace=True)
df["TT4"].replace({"?": 0}, inplace=True)
df["T4U"].replace({"?": 0}, inplace=True)
df["FTI"].replace({"?": 0}, inplace=True)
df["TBG"].replace({"?": 0}, inplace=True)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
df.TargetClass=le.fit_transform(df["TargetClass"])
df.Sex=le.fit_transform(df["Sex"])
df.on_thyroxine=le.fit_transform(df["on_thyroxine"])
df.query_on_thyroxine=le.fit_transform(df["query_on_thyroxine"])
df.on_antithyroid_medication=le.fit_transform(df["on_antithyroid_medication"])
df.thyroid_surgery=le.fit_transform(df["thyroid_surgery"])
df.query_hypothyroid=le.fit_transform(df["query_hypothyroid"])
df.query_hyperthyroid=le.fit_transform(df["query_hyperthyroid"])
df.pregnant=le.fit_transform(df["pregnant"])
df.sick=le.fit_transform(df["sick"])
df.tumor=le.fit_transform(df["tumor"])
df.lithium=le.fit_transform(df["lithium"])
df.goitre=le.fit_transform(df["goitre"])
df.TSH_measured=le.fit_transform(df["TSH_measured"])
df.T3_measured=le.fit_transform(df["T3_measured"])
df.TT4_measured=le.fit_transform(df["TT4_measured"])
df.FTI_measured=le.fit_transform(df["FTI_measured"])
df.TBG_measured=le.fit_transform(df["TBG_measured"])
df.T4U_measured=le.fit_transform(df["T4U_measured"])

x=df.iloc[0:, 1:26]

y=df.iloc[0:, 0]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,train_size=0.7)
print(Y_test)
# print("*************************** LogisticRegression *****************************")
# from sklearn.linear_model import LogisticRegression
#
# lr=LogisticRegression()
# modelLR=lr.fit(X_train,Y_train)
# pickle.dump(modelLR,open("hyp.pkl","wb"))
#
# prediction_lr = modelLR.predict(X_test)
#
# print("====================Prediction Of model=================")
# print(prediction_lr)
# print("====================ACtual Answers=================")
# print(Y_test)
#
#
# from sklearn.metrics import accuracy_score
# # =====================ACCUARACY===========================
# print("=====================Training Accuarcy=============")
# trac=lr.score(X_train,Y_train)
# trainingAccLR=trac*100
# print(trainingAccLR)
# print("====================Testing Accuracy============")
# teacLr=accuracy_score(Y_test,prediction_lr)
# testingAccLR=teacLr*100
# print(testingAccLR)

print("*************************** KNeighborsClassifier *****************************")
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
pickle.dump(knn,open("hyp.pkl","wb"))
prediction_knn = knn.predict(X_test)

print("====================Prediction Of model=================")
print(prediction_knn)
print("====================ACtual Answers=================")
print(Y_test)
# =====================ACCUARACY===========================
from sklearn.metrics import accuracy_score
print("=====================Training Accuarcy=============")
trac=knn.score(X_train,Y_train)
trainingAcc=trac*100
print(trainingAcc)
print("====================Testing Accuracy============")
teac=accuracy_score(Y_test,prediction_knn)
testingAcc=teac*100
print(testingAcc)