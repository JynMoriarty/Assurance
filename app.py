import pandas as pd
import streamlit as st
from PIL import Image
import pickle
image = Image.open('logo.jpg')
import numpy as np
import pydeck as pdk
p1 = open('modelRid.pkl', 'rb') 
ridge_model = pickle.load(p1)
p2 = open("modellr.pkl","rb")
lr_model = pickle.load(p2)
p3 = open("modelLasso.pkl","rb")
lasso_model = pickle.load(p3)
p4= open("modelEN.pkl","rb")
elastic_model = pickle.load(p4)
st.set_page_config(page_title="Assur'Aimant")
st.title('Prédiction des primes assurances')
regression=st.sidebar.selectbox("Choissisez l'algorithme de régression",("Linear","Ridge","Lasso","ElasticNet"))
st.image(image, caption='Votre assurance préférée')

bmi = st.number_input('Entrez votre bmi : ',12.00,70.00,18.50)
children = st.number_input("Entrez votre nombre d'enfants : ",0,10,0)
smoker = st.radio('Est ce que vous fumez ?',['yes','no'],horizontal=True)
sex = st.selectbox('Sexe :',['male','female'])
region = st.selectbox('Régions :',['southeast','southwest','northwest','northeast'],1)
age = st.slider('Age :',18,100,18,1)


liste=[bmi,children,smoker,sex,region,age]
columns = ['bmi','children','smoker','sex','region','age']





prediction = pd.DataFrame(np.array(liste).reshape(1,-1),columns=columns)
if st.button("Prédire"):
    if regression == "Linear":
        
        price = int(lr_model.predict(prediction))
        st.success("Votre prime d'assurance est de {} $".format(price))
    elif regression == "Ridge":
        price = int(ridge_model.predict(prediction))
        st.success("Votre prime d'assurance est de {} $".format(price))
    elif regression == "Lasso":
        price = int(lasso_model.predict(prediction))
        st.success("Votre prime d'assurance est de {} $".format(price))
    else:
        price = int(elastic_model.predict(prediction))
        st.success("Votre prime d'assurance est de {} $".format(price))