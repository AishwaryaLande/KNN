import pandas as pd
import numpy as np
glassData=pd.read_csv("C:\\Users\\ADMIN\\Desktop\\Data_Science_Assig\\KNN\\glass.csv")
glassData.columns
glassData.describe()
glassData.Type.value_counts()
len(glassData.columns)
glassData.drop_duplicates(keep='first',inplace=True)

glassData.isnull().sum()
glassData.shape

import matplotlib.pyplot as plt
import seaborn as sns
glassData.columns

plt.hist(glassData.RI)
plt.hist(np.log(glassData.RI)) # to normalize

plt.hist(glassData.Na)
plt.hist(np.log(glassData.Na)) # to normalize

plt.hist(glassData.Mg)
#plt.hist(np.log(glassData.Mg)) # to normalize

plt.hist(glassData.Al)
plt.hist(np.log(glassData.Al)) # to normalize

plt.hist(glassData.Si)
#plt.hist(np.log(glassData.Si)) # to normalize

plt.hist(glassData.K)
#plt.hist(np.log(glassData.K)) # to normalize

plt.hist(glassData.Ca)
plt.hist(np.log(glassData.Ca)) # to normalize

plt.hist(glassData.Ba)
#plt.hist(np.log(glassData.Ba)) # to normalize

plt.hist(glassData.Fe)
#plt.hist(np.log(glassData.Fe)) # to normalize

plt.hist(glassData.Type)
plt.hist(np.log(glassData.Type)) # to normalize

sns.boxplot(glassData.RI)
sns.boxplot(glassData.Na)
sns.boxplot(glassData.Mg)
sns.boxplot(glassData.Al)
sns.boxplot(glassData.Si)
sns.boxplot(glassData.K)
sns.boxplot(glassData.Ca)
sns.boxplot(glassData.Ba)
sns.boxplot(glassData.Fe)
sns.boxplot(glassData.Type)

sns.pairplot((glassData),hue='Type')

corr = glassData.corr()
corr
sns.heatmap(corr,annot=True)

from sklearn.model_selection import train_test_split
train,test = train_test_split(glassData,test_size = 0.3)
trainX = train.iloc[:,0:8]
trainY = train.iloc[:,9]
testX = test.iloc[:,0:8]
testY = test.iloc[:,9]

from sklearn.neighbors import KNeighborsClassifier as KNN
model1 = KNN(n_neighbors=2).fit(trainX,trainY)
model1_pred = model1.predict(trainX)
accurancy_m1 = nm.mean(model1_pred == trainY)
accurancy_m1

model2 = KNN(n_neighbors=5).fit(trainX,trainY)
model2_pred = model2.predict(trainX)
accurancy_m2 = nm.mean(model2_pred == trainY)
accurancy_m2

accurancy_ar=[];

for i in range(5,50):
    modeli = KNN(n_neighbors=i).fit(trainX,trainY)
    modeli_train_pred = modeli.predict(trainX)
    modeli_test_pred = modeli.predict(testX)
    train_acc = nm.mean(modeli_train_pred == trainY) 
    test_acc = nm.mean(modeli_test_pred == testY) 
    accurancy_ar.append([train_acc,test_acc])

import matplotlib.pyplot as plt
plt.plot(nm.arange(5,50),[i[0] for i in accurancy_ar],"bo-")
plt.plot(nm.arange(5,50),[i[1] for i in accurancy_ar],"ro-")
plt.legend(["train","test"])
