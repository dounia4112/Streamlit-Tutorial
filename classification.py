import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



def app():
    st.title("✅ Classification")

    if 'uploaded_file' in st.session_state:
        df = st.session_state['uploaded_file']

        st.write('Data Loaded for classification')
        st.write(df.head())

        target_column = df.columns[-1]
        st.markdown(f'''
            <div style= "background-color: #f9f9f9; border-radius: 10px; padding: 10px, text-align: center;">
                <h3 style = "color: blue">Target Column: <span style="font-weight: bold"> {target_column} </span></h3>
            </div>
        ''', unsafe_allow_html=True)

        X = df.drop(columns=[target_column])
        y = df[target_column]

        test_size = st.slider('Test Size (as%)', min_value=10, max_value=50,value=20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        algorithm = st.selectbox('Choose Algorithm',
                                    options= ['Logistic Regression', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors'])
        
        if algorithm== 'Logistic Regression':
            cls = LogisticRegression()
        elif algorithm == 'Decision Tree':
            criterion = st.radio('Type of Criterion', options=['gini', 'entropy'])
            max_depth = st.slider('Maximum Depth', min_value=5, max_value=20, value=5)
            min_samples_leaf = st.slider('Minimum Samples Leaf', 10,50,20)
            cls = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        elif algorithm == 'Random Forest':
            n_estimators = st.slider('Number of Estimators', min_value=10, max_value=50, value=10)
            max_depth = st.slider('Maximum Depth', min_value=2, max_value=20, value=5)
            cls = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        else:
            n_neighbors = st.slider('Number of Neighbors', min_value=2, max_value=50, value=4)
            metric = st.selectbox('Metric', options=['euclidean', 'minkowski', 'manhattan'])
            cls = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        
        if st.button('Train Model'):
            cls.fit(X_train, y_train)
            y_pred = cls.predict(X_test)
            tab1, tab2 = st.tabs(["Metrics", "Confusion Matrix"])

            with tab1:
                st.write("### Performence Metrics")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format(precision=2))
            
            with tab2:
                cm = confusion_matrix(y_test, y_pred)
                st.write("### Confusion Matrix")
                fig_cm , ax = plt.subplots()
                vis = ConfusionMatrixDisplay(cm, display_labels=cls.classes_)
                vis.plot(ax=ax, cmap="Blues", colorbar=False)
                st.pyplot(fig_cm)

    else:
        st.write('No Data Found. Please upload a data on the EDA page first.')







# if __name__ == "__main__":
#     app()