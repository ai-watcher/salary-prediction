import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
label_country = data["label_country"]
label_education = data["label_education"]
label_gender = data["label_gender"]


def show_predict_page():
    st.title("Developer Salary Prediction")

    st.write("""
    ## We need some information to predict the Salary
    """)

    countries = (
            'Argentina',
            'Australia',
            'Austria',
            'Bangladesh',
            'Belgium',
            'Brazil',
            'Bulgaria',
            'Canada',
            'Chile',
            'China',
            'Colombia',
            'Croatia',
            'Czech Republic',
            'Denmark',
            'Egypt',
            'Finland',
            'France',
            'Germany',
            'Greece',
            'Hungary',
            'India',
            'Indonesia',
            'Iran',
            'Ireland',
            'Israel',
            'Italy',
            'Japan',
            'Lithuania',
            'Malaysia',
            'Mexico',
            'Netherlands',
            'New Zealand',
            'Nigeria',
            'Norway',
            'Other',
            'Pakistan',
            'Philippines',
            'Poland',
            'Portugal',
            'Romania',
            'Russian Federation',
            'Serbia',
            'Singapore',
            'Slovenia',
            'South Africa',
            'Spain',
            'Sri Lanka',
            'Sweden',
            'Switzerland',
            'Taiwan',
            'Turkey',
            'Ukraine',
            'United Kingdom of Great Britain and Northern Ireland',
            'United States of America',
            'Viet Nam'  
    )

    education = (
        'Less than a Bachelors',
        'Bachelor’s degree',
        'Master’s degree',
        'Post grad'
    )

    gender = (
        'Man',
        'Woman',
        'Others'
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education", education)
    gender = st.selectbox("Gender", gender)
    experience = st.slider("Years of Experience", 0, 40, 2)

    sal = st.button("Calculate Salary")
    if sal:
        X = np.array([[country, education, experience, gender]])
        X[:, 0] = label_country.transform(X[:,0])
        X[:, 1] = label_education.transform(X[:,1])
        X[:, 3] = label_gender.transform(X[:,3])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
