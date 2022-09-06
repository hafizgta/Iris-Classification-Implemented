import keras #library for neural network
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 

import pandas as pd #loading data in table form  
import seaborn as sns #visualisation 
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra

from sklearn.preprocessing import normalize #machine learning algorithm library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import argparse
from uuid import uuid1 as uuid
from utils import update_model_metadata,upload_model_to_object_storage,call_for_model_hotswap
import os
import joblib


storage_target=os.getenv("STORAGE_TARGET","s3://iris_classification")

#  Load the dataset, which contains the data points(sepal length, petal length, etc) and corresponding labels(type of iris)
def load_dataset(csv_file):
    iris_dataset=pd.read_csv(csv_file)

    iris_dataset.loc[iris_dataset["species"]=="setosa","species"]=0
    iris_dataset.loc[iris_dataset["species"]=="versicolor","species"]=1
    iris_dataset.loc[iris_dataset["species"]=="virginica","species"]=2

    return iris_dataset

def load_dataset_from_query():
    # iris_dataset.loc[iris_dataset["species"]=="setosa","species"]=0
    # iris_dataset.loc[iris_dataset["species"]=="versicolor","species"]=1
    # iris_dataset.loc[iris_dataset["species"]=="virginica","species"]=2
    pass

def split_dataset(iris_dataset):
    # Break the dataset up into the examples (X) and their labels (y)
    X = iris_dataset.iloc[:, 0:4].values
    y = iris_dataset.iloc[:, 4].values
    X=normalize(X,axis=0)

    # Split up the X and y datasets randomly into train and test sets
    # 20% of the dataset will be used for the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=31)

    #Change the label to one hot vector
    '''
    [0]--->[1 0 0]
    [1]--->[0 1 0]
    [2]--->[0 0 1]
    '''
    y_train=np_utils.to_categorical(y_train,num_classes=3)
    y_test=np_utils.to_categorical(y_test,num_classes=3)

    return y_train, y_test, X_train, X_test



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        nargs="?",
        default="https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv",
        help="define which iris dataset is used, disabled if query is defined"
    )
    
    parser.add_argument(
        "--query",
        type=bool,
        nargs="?",
        default=False,
        help="use predefined query to get dataset"
    )

    parser.add_argument(
        "--batch",
        type=int,
        nargs="?",
        default=20,
        help="batch size"
    )

    parser.add_argument(
        "--epoch",
        type=int,
        nargs="?",
        default=10,
        help="number of epochs"
    )

    parser.add_argument(
        "--output",
        type=str,
        nargs="?",
        default="./inferencing/models/",
        help="location of where to save model"
    )
    parser.add_argument(
        "--modelname",
        type=str,
        nargs="?",
        default="model",
        help="define model name"
    )


    opt = parser.parse_args()
    model_id=uuid()
    # load and split dataset
    
    if opt.query:
        iris_dataset = load_dataset_from_query()
    else:
        iris_dataset = load_dataset(opt.dataset)

    y_train, y_test, X_train, X_test= split_dataset(iris_dataset)

    model = Sequential()
    # Adding the input layer and the first hidden layer
    model.add(Dense(1000,input_dim=4,activation='relu'))
    model.add(Dense(50,activation='relu'))
    #Protects against overfitting
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(3,activation='softmax'))
    # Compiling the ANN
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    # Fitting the ANN to the Training set
    model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=opt.batch,epochs=opt.epoch,verbose=1)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    prediction=model.predict(X_test)
    length=len(prediction)
    y_label=np.argmax(y_test,axis=1)
    predict_label=np.argmax(prediction,axis=1)
    #how times it matched/ how many test cases
    accuracy=np.sum(y_label==predict_label)/length * 100 
    print("Accuracy of the dataset",accuracy )
    model.save(f'{opt.output}/{opt.modelname}.h5')
if __name__=="__main__":
    main()