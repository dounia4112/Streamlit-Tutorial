import streamlit as st

import EDA, classification

st.sidebar.title('Data Analysis')

page = st.sidebar.radio('Select A Page', options=['EDA', "Classification"])

if page == "EDA":
    EDA.app()
elif page == "Classification":
    classification.app()