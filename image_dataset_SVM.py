import numpy as np # linear algebra
import pandas as pd
#from subprocess import check_output
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv("images_data.csv")

#preprocessing
dataset["target"]=dataset["target"].map({'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9})

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#splitting dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=42)


# Create a classifier: a support vector classifier
from sklearn import svm
classifier = svm.SVC(kernel='rbf',random_state=42)
model=classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)



#Model Accuracy.
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
model_accurcy=accuracy_score(y_test, y_pred)
print(f'Model Accuracy : {model_accurcy}%')

#ROC curve.
from sklearn.metrics import roc_curve
from matplotlib import pyplot
ns_probs = [0 for _ in range(len(y_test))]
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
#from sklearn.metrics import plot_roc_curve
#plot_roc_curve(classifier, X_test, y_test)

# summarize history for loss


