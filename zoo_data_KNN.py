import pandas as pd
import numpy as nm
zooData = pd.read_csv("C:/Users/ADMIN/Desktop/Data_Science_Assig/KNN/Zoo.csv")
zooData.columns
zooData.describe()
zooData.type.value_counts()
len(zooData.columns)

zooData.drop_duplicates(keep='first',inplace=True)

zooData.isnull().sum()
zooData.shape
zooData.columns

import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(zooData.legs)     #other columns have values in binary that is 0 and 1 
plt.hist(zooData.type)

sns.boxplot(zooData.legs)
sns.boxplot(zooData.type)

sns.pairplot((zooData),hue='type')
corr=zooData.corr()
corr
sns.heatmap(corr,annot=True)

from sklearn.model_selection import train_test_split
train_data_zoo,test_data_zoo = train_test_split(zooData,test_size=0.3)
train_data_zoo_X = train_data_zoo.iloc[:,1:17]
train_data_zoo_Y = train_data_zoo.iloc[:,17]
test_data_zoo_X = test_data_zoo.iloc[:,1:17]
test_data_zoo_Y = test_data_zoo.iloc[:,17]

from sklearn.neighbors import KNeighborsClassifier as KNN
model1 = KNN(n_neighbors=5).fit(train_data_zoo_X,train_data_zoo_Y)
predict_val = model1.predict(train_data_zoo_X)
acc_model1 = nm.mean(predict_val == train_data_zoo_Y)
acc_model1

acc_arr = [];

for i in range(3,60):
    modeln = KNN(n_neighbors=i).fit(train_data_zoo_X,train_data_zoo_Y)
    predict_val_train = modeln.predict(train_data_zoo_X)
    predict_val_test = modeln.predict(test_data_zoo_X)
    acc_model_train = nm.mean(predict_val_train == train_data_zoo_Y)
    acc_model_test = nm.mean(predict_val_test == test_data_zoo_Y)
    acc_arr.append([acc_model_train,acc_model_test])
    
acc_arr   

import matplotlib.pyplot as plt
plt.plot(nm.arange(3,60),[i[0] for i in acc_arr],"bo-")    
plt.plot(nm.arange(3,60),[i[1] for i in acc_arr],"ro-") 
plt.legend(["train","test"])