import pandas as pd
import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components

# Page Configuration:

st.set_page_config(page_title = "HCP DashBoard", page_icon = ":ledger:", layout = "wide")

# Header Section:

st.subheader("High Commission for Planning of Morocco")

st.markdown("Adib HABBOU")

st.title("Data Analysis of Moroccan Newspapers")

# Data Set:

st.header("Data Set Sample")

df_all = pd.read_csv("morocco_world_news_articles_v1.csv")

df_all.drop(columns=['Unnamed: 0'], inplace = True)

st.dataframe(df_all)

# Topic Model:

st.header("Topic Modeling")

lda = open("lda.html", 'r', encoding='utf-8')

source_code = lda.read() 

components.html(source_code, width = 1700, height = 800)

# News Categories:

st.header("News Categories")

df = pd.read_csv("morocco_world_news_articles.csv")

fig = px.pie(df, names = "category")

st.plotly_chart(fig, use_container_width = True)

# News Authors:

st.header("News Authors")

author = df_all.groupby('author').count()

author.drop(author[author.title < 6].index, inplace = True)

author.drop(["lead", "date", "content"], axis = 1, inplace = True)

fig = px.bar(author)

st.plotly_chart(fig, use_container_width = True)

# Classification Accuarcy:

st.header("Classification Accuarcy")

acc = pd.DataFrame(['K Nearest Neighbor', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine', 'Stochastic Gradient Descent', 'Multi-Layer Perceptron'], [71, 85, 72, 84, 85, 85, 83])

fig = px.scatter(acc, color = 'value', labels = {'value': 'Machine Learning Model', 'index': 'Accuracy Score (%)'})

fig.update_traces(marker = dict(size = 20, line = dict(width = 1, color = "DarkSlateGrey")), selector = dict(mode = "markers"))

st.plotly_chart(fig, use_container_width = True)
