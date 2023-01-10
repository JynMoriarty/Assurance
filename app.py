import pandas as pd
import streamlit as st
from PIL import Image
import pickle
image = Image.open('logo.jpg')
import numpy as np
import pydeck as pdk

p2 = open("modellr.pkl","rb")
lr_model = pickle.load(p2)
p4= open("modelEN.pkl","rb")
elastic_model = pickle.load(p4)
st.set_page_config(page_title="Assur'Aimant")
st.title('Prédiction des primes assurances')
regression=st.sidebar.selectbox("Choissisez l'algorithme de régression",("Linear","Ridge","Lasso","ElasticNet"))
st.image(image, caption='Votre assurance préférée')

taille = st.number_input("Entrez votre taille (en cm)",140,200,160)
poids = st.number_input('Entrez votre poids (en kg)',20,200,50 )
children = st.number_input("Entrez votre nombre d'enfants : ",0,10,0)
smoker = st.radio('Est ce que vous fumez ?',['yes','no'],horizontal=True)
sex = st.selectbox('Sexe :',['male','female'])
region = st.selectbox('Régions :',['southeast','southwest','northwest','northeast'],1)
age = st.slider('Age :',18,100,18,1)
bmi_int = round(poids/(taille*(10**(-2)))**(2),2)
if bmi_int > 15 and bmi_int<=24:
    bmi = 'normal'
elif bmi_int > 24 and bmi_int <= 30:
    bmi = 'surpoids'
elif bmi_int > 30 and bmi_int <=40 :
    bmi = 'obèse'
elif bmi_int > 40 and bmi_int <= 54 :
    bmi = 'obésité morbide'
st.write('votre bmi est de',str(bmi_int),' et vous etes de la catégorie ',str(bmi) )

liste=[bmi,children,smoker,sex,region,age]
columns = ['bmi','children','smoker','sex','region','age']





prediction = pd.DataFrame(np.array(liste).reshape(1,-1),columns=columns)
if st.button("Prédire"):
    if regression == "Linear":
        price = int(lr_model.predict(prediction))
        st.success("Votre prime d'assurance est de {} $".format(price))
    else:
        price = int(elastic_model.predict(prediction))
        st.success("Votre prime d'assurance est de {} $".format(price))
