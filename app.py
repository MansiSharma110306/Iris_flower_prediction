import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load model and data
iris = load_iris()
X = iris.data
y = iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# UI
st.title("Iris Flower Species Predictor")
st.write("Enter flower measurements below:")

sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted species: **{iris.target_names[prediction]}**")
#streamlit run app.py