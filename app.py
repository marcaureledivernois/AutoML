import streamlit as st
import pandas as pd
import os
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import pickle
import numpy as np

with st.sidebar:
    st.image("https://learn.g2.com/hubfs/G2CM_FI416_Learn_Article_Images-%5BRegression_Analysis%5D_V1a.png")
    st.title("Automatic Machine Learning Tool")
    choice = st.radio("Steps", ["Upload Data", "Exploratory Data Analysis", "Automatic ML","Predictions", "Download Best Model"])
    st.info("This tool automatically explores data, builds an automatic ML pipline and let you download the best model at the end.")

if os.path.exists('sourcedata.csv'):
    df = pd.read_csv('sourcedata.csv', index_col=None)
    # config_dict = {'target' : 'Unknown', 'problem_type':'Unknown'}
    # with open('config.pkl', 'wb') as f:
    #     pickle.dump(config_dict, f)

if choice == "Upload Data":
    st.info("Upload your dataset as csv format here. Rows are observations, Columns are features.")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv('sourcedata.csv', index=False)
        st.dataframe(df)
        pass

if choice =="Exploratory Data Analysis":
    st.title('Automated Exploratory Data Analysis')
    if st.button("Run analysis"):
        profile_report = df.profile_report()
        st_profile_report(profile_report)

if choice == "Automatic ML":
    st.title('Machine Learning')
    target = st.selectbox("Select the target", df.columns)
    problem_type = st.selectbox("Select the problem type", ["Classification", "Regression"])
    if problem_type == "Classification":
        from pycaret.classification import setup, compare_models, pull, save_model, load_model
    if problem_type == "Regression":
        from pycaret.regression import setup, compare_models, pull, save_model, load_model

    if st.button("Train models"):
        d= {}
        # Update the values
        d['target'] = target
        d['problem_type'] = problem_type

        # Save
        with open('config.pkl', 'wb') as f:
            pickle.dump(d, f)

        setup(df, target=target, session_id = 42)
        setup_df = pull()
        st.info("Below is the ML Experiment settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("Below is the comparison of ML models tested")
        st.dataframe(compare_df)
        st.info("The best model for your problem is :")
        st.info(best_model)
        save_model(best_model, 'best_model')

if choice == "Predictions":
    with open('config.pkl', 'rb') as f:
        d = pickle.load(f)
    target = d['target']
    problem_type = d['problem_type']

    features = {}
    for col, dtype in zip(df.columns, df.dtypes):
        if col != target:
            if dtype == 'object':
                features[col] = st.text_input("Enter a value for " + col, df[col].value_counts().idxmax())
            else:
                features[col] = st.number_input("Enter a value for " + col, value = df[col].value_counts().idxmax())

    ordered_features = {k : features[k] for k in df.columns if k != target}

    if st.button("Run prediction"):
        if problem_type=="Regression":
            from pycaret.regression import load_model
        if problem_type == "Classification":
            from pycaret.classification import load_model, predict_model
        best_model = load_model('best_model')
        prediction = predict_model(best_model, data=pd.DataFrame(ordered_features, index=[0])).loc[0,'prediction_label']
        probability = predict_model(best_model, data=pd.DataFrame(ordered_features, index=[0])).loc[0,'prediction_score']

        st.info('Prediction : ' +  str(prediction) + ' (with probability : '  +  "{0:.0%}".format(probability) + ')')


if choice == "Download Best Model":
    with open("best_model.pkl", "rb") as f:
        st.download_button("Dowload the best model", f, "best_model.pkl")