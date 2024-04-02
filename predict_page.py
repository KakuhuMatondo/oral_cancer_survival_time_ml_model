import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn


def load_model():
    with open('model.pkl', 'rb') as file:
        d = pickle.load(file)
    return d


data = load_model()

model = data['model']


def show_predict_page():
    st.title("Cancer Patient Survival Time Prediction")
    st.write("""### We need some information to predict the Survival Time""")

    Gender = ('Male', 'Female')
    TumorStage = (10, 20, 30, 40)

    gender = st.radio("Gender", Gender)
    tumorstage = st.selectbox("Tumor Stage", TumorStage)
    age = st.slider("Age", 10, 90, 1)
    genemutations = st.number_input("Number of Gene Mutations", 0, 11000,step=1,
                                    help="Number of Mutated Genes Permitted 0 - 11,000")
    ok = st.button("Calculated Estimated Survival time in days")
    if ok:
        gender_encoded = 0 if gender == 'Male' else 1  # Encoding gender as numerical value

        # Create feature vector

        feature_list=[gender_encoded, age, genemutations, tumorstage]
        print(feature_list)
        f = np.array(feature_list,dtype=float).reshape(1, -1)

        # Make prediction
        prediction = model.predict(f)
        formatted_prediction = "{:,.0f}".format(prediction[0])
        st.subheader(f"The estimated number of survival days is {formatted_prediction} ")

