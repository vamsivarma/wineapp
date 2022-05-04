#!/usr/bin/env python
# coding: utf-8

# In[5]:


#import libraries
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
#app heading
st.title("""
Wine Quality Prediction App
This app predicts the ***Wine Quality*** type using ML-RandomForestClassifier!
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')
  
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58,0.52)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        residual_sugar=st.sidebar.slider('residual sugar',0.9,15.5,2.53)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6, 0.08)
        free_sulfur_dioxide=st.sidebar.slider('free sulfur dioxide',0.01,0.61,0.087)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0, 46.0)
        density=st.sidebar.slider('density',0.99,1.0,0.99)
        pH=st.sidebar.slider('pH',2.74,4.01,3.33)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
       
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'residual_sugar':residual_sugar,
                'chlorides': chlorides,
                'free_sulfur_dioxide':free_sulfur_dioxide,
              'total_sulfur_dioxide':total_sulfur_dioxide,
                'density':density,
                'pH':pH,
              'sulphates':sulphates,
                'alcohol':alcohol}
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)
#reading csv file
data=pd.read_csv("winequality-red.csv")
st.write(data.head(2))
X =np.array(data[['fixed acidity', 'volatile acidity' , 'citric acid', 'residual sugar', 'chlorides' ,'free sulfur dioxide', 'total sulfur dioxide' , 'density','pH', 'sulphates','alcohol']])
Y = np.array(data['quality'])
Y = [1 if a_ >= 7 else 0 for a_ in Y]

#Y=data['quality'].apply(lambda y_value: 1 if y_value>=7 else 0)
#random forest model
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=3)
#st.write(Y.shape,Y_train.shape,Y_test.shape)
model= RandomForestClassifier()
model.fit(X_train, Y_train)
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
st.subheader('Accuracy of the model:')
st.write(test_data_accuracy)
def predict():
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    if (prediction[0]==1):
        Quality='Good Quality wine'
    else:
        Quality='Bad Quality Wine'    
    st.header('Quality')
    st.write(Quality)
    st.write(prediction_proba)
    
if st.button('Predict'):
    predict()


# In[ ]:




