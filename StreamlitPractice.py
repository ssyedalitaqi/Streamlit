import streamlit as st
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_name=st.sidebar.selectbox("Select the  data",("Breast Cancer","Iris","Wine"))
algorithms=st.sidebar.selectbox("Select the algorithm",("KNN","SVM"))


def get_data(name):
    data=None
    if name=="Breast Cancer":
        data=datasets.load_breast_cancer()
    elif name=="Iris":
        data=datasets.load_iris()
    else: 
        data=datasets.load_wine()
    x=data.data
    y=data.target
    return x,y


x,y=get_data(data_name)
st.dataframe(x)

# for shape of data
st.write("the shape of data is ",x.shape)

# for unique value of y
st.write("the unique target",len(np.unique(y)))

# for data visualization
sns.boxplot(data=x,orient="h")
st.pyplot()


plt.hist(x)
st.pyplot()

# building the Algorithm
def get_parameter(name_of_clf):
    params=dict()
    if name_of_clf=="SVM":
        c=st.sidebar.slider("Value of c",0.01,15.0)
        params['C']=c
    else:
        name_of_clf=="KNN"
        k=st.sidebar.slider("value of k",1,15)
        params['k']=k
    return params
params=get_parameter(algorithms)  

# Accesing our classifier
def get_classifier(name_of_clf,params):
    clf=None
    if name_of_clf=="SVM":
        clf=SVC(C=params['C'])
    else:
        clf=KNeighborsClassifier(n_neighbors=params["k"])   
    return clf
clf=get_classifier(algorithms,params) 


# Training the data onto algorithm
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
st.write("The prediction of the above data is ",y_predict)
accuracy=accuracy_score(y_test,y_predict)
st.write("Classifier name",algorithms)
st.write("the accuracy of the model is ",accuracy)