#import pandas
import pandas as pd
#import numpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb



dataset=pd.read_csv("images_data.csv")

#preprocessing
dataset["target"]=dataset["target"].map({'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9})

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=42)




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
classifier.add(Dense(units=512,kernel_initializer='he_uniform',activation='relu',input_dim=64))
#second hidden layer
classifier.add(Dense(units=256,kernel_initializer='he_uniform',activation='relu'))
#second hidden layer
classifier.add(Dense(units=128,kernel_initializer='he_uniform',activation='relu'))
# last layer or output layer
classifier.add(Dense(units=10,kernel_initializer='he_uniform',activation='softmax'))


#classifier.summary()

classifier.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model=classifier.fit(X_train,y_train,batch_size=100,epochs=200)

y_pred=classifier.predict(X_test)
#y_pred=(y_pred>.5)

y_pred=pd.DataFrame(y_pred).idxmax(axis=1)

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
model_accurcy=accuracy_score(y_test,y_pred)
#confusion matrix.
print("confusion matrix : \n",cm)

#Model Accuracy.
print(f'Model Accuracy : {model_accurcy}%')

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
