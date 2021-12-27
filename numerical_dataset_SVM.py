import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
dataset=pd.read_csv("data.csv")

#print(dataset.head())
#print(dataset["diagnosis"].value_counts())
#sns.countplot(dataset["diagnosis"],label="count")

"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
"""
from sklearn.preprocessing import LabelEncoder#categorical data convert numbers
labelencoder=LabelEncoder()
dataset.iloc[:,1]=labelencoder.fit_transform(dataset.iloc[:,1].values)
print(dataset.iloc[:,1])
#sns.pairplot(dataset.iloc[:,1:6])

X=dataset.iloc[:,2:32].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split #test_size is size of test data and train data will be (1-test size)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler #
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


#model SVM
from sklearn.svm import SVC
classifier=SVC(kernel="rbf",random_state=42)
classifier.fit(X_train,y_train)

#
y_pred=classifier.predict(X_test)
    


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,y_test)
model_accurcy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][1]+cm[1][0])
model_accurcy=model_accurcy*100
#confusion matrix.
print("confusion matrix : \n",cm)

#Model Accuracy.
print(f'Model Accuracy : {model_accurcy}%')

#ROC curve.
from sklearn.metrics import plot_roc_curve
plot_roc_curve(classifier, X_test, y_test)

#Loss curve.
#????




