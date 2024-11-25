import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

def app():
    st.title("Exploratory Data Analysis (EDA)")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.write("### Dataset Preview")
        st.write(df.head())

        st.write("### Descriptive Statistics")
        st.write(df.describe())

        st.write("### Correlation Matrix")
        if st.checkbox("Show Correlation Matrix"):
            corr_matrix = df.corr(numeric_only=True)
            fig_corr = px.imshow(corr_matrix, text_auto=True)
            st.plotly_chart(fig_corr)

        st.write("### Advanced Visualizations")
        st.write("Select Columns for visualization:")
        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols) >= 2:
            x_axis = st.selectbox("X-Axis", options=num_cols)
            y_axis = st.selectbox("Y-Axis", options=[col for col in num_cols if x_axis != col])
            fig_scatter = px.scatter(df, x=x_axis, y=y_axis, title="Scatter Plot")
            st.plotly_chart(fig_scatter)
        else:
            st.warning("Not Enough Columns for Scatter Plot")

        st.write("### Data Distribution")
        num_cols_without_target = df.drop(columns=df.columns[-1]).select_dtypes(include='number').columns
        feature = st.selectbox("Feature", options=num_cols_without_target)
        fig_hist, ax = plt.subplots()
        ax.hist(x=df[feature],bins=20, edgecolor="black")
        ax.set_title(f"Histogram of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        st.pyplot(fig_hist)




if __name__ == "__main__":
    app()