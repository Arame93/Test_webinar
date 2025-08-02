import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")

st.title("Prédiction de l'espéce Iris")
st.write("Entrez les caractéristiques de la fleur pour prédire son espéce")

sepal_length = st.slider("la longueur du sépale (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("largeur du sépale (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("la longueur du pétale (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("la largeur du pétale (cm)", 0.1, 2.5, 1.0)

input_data = np.array([[sepal_length, sepal_width,petal_length, petal_width]]) 

if st.button("Prédire"):
    prediction = model.predict(input_data)[0]
    species = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"L'espéce prédite est: {species[prediction]} ")

