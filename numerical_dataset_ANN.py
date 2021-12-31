#import pandas
import pandas as pd
#import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


dataset = pd.read_csv('data.csv')
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
dataset.iloc[:,1]=labelencoder.fit_transform(dataset.iloc[:,1].values)
print(dataset.iloc[:,1])


X=dataset.iloc[:,2:32].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)



#importing keras
import keras

#importing sequential module
from keras.models import Sequential
# import dense module for hidden layers
from keras.layers import Dense
#importing activation functions
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout



#creating model
classifier = Sequential()

#first hidden layer
classifier.add(Dense(units=9,kernel_initializer='he_uniform',activation='relu',input_dim=30))
#second hidden layer
classifier.add(Dense(units=9,kernel_initializer='he_uniform',activation='relu'))

# last layer or output layer
classifier.add(Dense(units=1,kernel_initializer='he_uniform',activation='sigmoid'))


#classifier.summary()

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model=classifier.fit(X_train,y_train,batch_size=100,epochs=500)



y_pred=classifier.predict(X_test)
y_pred=(y_pred>.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
model_accurcy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][1]+cm[1][0])
model_accurcy=model_accurcy*100
#confusion matrix.
print("confusion matrix : \n",cm)

#Model Accuracy.
print(f'Model Accuracy : {model_accurcy}%')

#ROC curve.
from sklearn.metrics import roc_curve
from matplotlib import pyplot
ns_probs = [0 for _ in range(len(y_test))]
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.show()

# list all data in history
print(model.history.keys())
# summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
