# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:34:51 2020

@author: ANKUR
"""

import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Machine Learning Web App")

st.write("""
         #Explore different classifiers
         
         Which one is the best?
    """)
    
dataset_name = st.sidebar.selectbox("Select dataset", ("Iris", "Breast Cancer", "Wine dataset"))    

classifier_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))    

def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()  
    else:
        data = datasets.load_wine()    
        
    X = data.data
    y = data.target
    return X, y        

X, y = get_dataset(dataset_name)
st.write("Shape of Dataset:", X.shape)
st.write("Number of classes:", len(np.unique(y)))


def add_parameter(classifier_name):
    params = dict()
    if classifier_name == 'KNN':
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    elif classifier_name == 'SVM':
        C = st.sidebar.slider("C", 0.01, 10.0)  
        params['C'] = C
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params
    
params = add_parameter(classifier_name) 
   
def get_classifier(classifier_name, params):
    if classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors= params["K"])
      
    elif classifier_name == 'SVM':
        clf = SVC(C = params['C'])  
        
    else:
       # max_depth = st.sidebar.slider("Max Depth", 2, 15)
       # n_estimators = st.sidebar.slider("n_estimators", 1, 100)
        clf = RandomForestClassifier(n_estimators = params['n_estimators'], max_depth = params['max_depth'], random_state=1234)
    
    return clf

clf = get_classifier(classifier_name, params)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234) 

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 

score = accuracy_score(y_test, y_pred)
    
st.write(f"classifier= {classifier_name}")
st.write(f"accuracy= {score}")

pca = PCA(2)
X_projected = pca.fit_transform(X)

x1 = X_projected[:,0]
x2 = X_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c= y, cmap = 'viridis')

plt.xlabel("principal component 1")
plt.ylabel("principal component 2")

plt.colorbar()

st.pyplot()



