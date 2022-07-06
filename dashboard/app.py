import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

# ---Page Configuration---

st.set_page_config(page_title="HCP DashBoard", page_icon=":ledger:", layout="wide")

# ---Header Section---

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("")
    with col2:
        logo_hcp = Image.open('image/logo hcp horizontal.png')
        st.image(logo_hcp, width=400)
    with col3:
        st.write("")
    st.title("Data Analysis of Moroccan Newspapers")

# ---Data Set Sample---

with st.container():
    st.header("Morocco World News Data Set Sample")
    df_all = pd.read_csv("csv/morocco_world_news_articles_v1.csv")
    df_all.drop(columns=['Unnamed: 0'], inplace=True)
    st.dataframe(df_all[:10])

# ---Topic Model Vis---

with st.container():
    st.header("Topic Modeling using Latent Dirichlet Allocation")
    lda = open("html/lda.html", 'r', encoding='utf-8')
    source_code = lda.read()
    components.html(source_code, width=1700, height=800)

# ---News Categories---

with st.container():
    st.header("Morocco World News Categories")
    df = pd.read_csv("csv/morocco_world_news_articles.csv")
    fig = px.pie(df, names="category")
    st.plotly_chart(fig, use_container_width=True)

# ---News Authors---

with st.container():
    st.header("Morocco World News Authors")
    author = df_all.groupby('author').count()
    author.drop(author[author.title < 6].index, inplace=True)
    author.drop(["lead", "date", "content"], axis=1, inplace=True)
    fig = px.bar(author)
    st.plotly_chart(fig, use_container_width=True)

# ---Classification Accuracy---

with st.container():
    st.header("Classification Models Accuracies")
    acc = pd.DataFrame(
        ['K Nearest Neighbor', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine',
         'Stochastic Gradient Descent', 'Multi-Layer Perceptron'], [71, 85, 72, 84, 85, 85, 83])
    fig = px.scatter(acc, color='value', labels={'value': 'Machine Learning Model', 'index': 'Accuracy Score (%)'})
    fig.update_traces(marker=dict(size=20, line=dict(width=1, color="DarkSlateGrey")), selector=dict(mode="markers"))
    st.plotly_chart(fig, use_container_width=True)

# ---Confusion Matrix---

with st.container():
    st.header("Confusion Matrix Logistic Regression")
    LRmat = Image.open('image/LRmat.png')
    st.image(LRmat, width=1200)

with st.container():
    st.header("Confusion Matrix Stochastic Gradient Descent")
    SGDmat = Image.open('image/SGDmat.png')
    st.image(SGDmat, width=1200)

with st.container():
    st.header("Confusion Matrix Support Vector Machine")
    SVMmat = Image.open('image/SVMmat.png')
    st.image(SVMmat, width=1200)

# ---Footer Section---


