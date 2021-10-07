import streamlit as st
import  numpy as np
import  pandas as pd
import matplotlib.pyplot as plt


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'


def clean_gender(x):
    if 'Man' in x:
        return 'Man'
    if 'Woman' in x:
        return 'Woman'
    return 'Others'

@st.cache
def load_data():
    df = pd.read_csv("Files/survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "Gender", "ConvertedCompYearly"]]
    df = df[df["ConvertedCompYearly"].notnull()]
    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedCompYearly"] <= 200000]
    df = df[df["ConvertedCompYearly"] >= 5000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df['Gender'] = df['Gender'].apply(clean_gender)
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    return df

df = load_data()



def show_explore_page():
    st.title("Explore Developer Engineer Salaries")

    st.write(
        """
    ### Stack Overflow Developer Survey 2021
    """
    )

    data = df["Country"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)
    
    st.write(
        """
    #### Mean Salary Based On Country
    """
    )

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

    """
    #### Mean Salary Based On Gender
    """
    data = df.groupby(["Gender"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)