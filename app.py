import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

st.set_page_config(layout="wide")
st.title("Lease-Up 10-Feature Insights Dashboard with RAG Q&A")

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
user_q = st.sidebar.text_area("Ask a question about your lease-up features:")
button = st.sidebar.button("Generate Answer")

@st.cache_data
def load_data():
    df = pd.read_csv("Market1_Final_Selected.csv")
    return df

if api_key:
    client = OpenAI(api_key=api_key)
    df = load_data()

    def answer_query(query):
        sample_context = df.head(20).to_string()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a lease-up assistant with 10 feature context. Use only the given data."},
                {"role": "user", "content": f"Data:\n{sample_context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content

    if button and user_q:
        reply = answer_query(user_q)
        st.sidebar.subheader("Answer")
        st.sidebar.write(reply)

    st.header("10 Features Overview")

    # Display each feature with an appropriate plot
    fig1 = px.histogram(df, x="delivery_year", title="Delivery Year Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(df["Submarket"].value_counts().reset_index(), x="index", y="Submarket", title="Submarket Counts")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df, x="leaseup_time", title="Lease-Up Time Distribution")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(df, x="effective_rent_delivery", y="effective_rent_leaseup", title="Delivery vs Lease-Up Rent")
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.histogram(df, x="effective_rent_growth", title="Effective Rent Growth Distribution")
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.pie(df, names="negative_growth", title="Negative Growth Proportion")
    st.plotly_chart(fig6, use_container_width=True)

    fig7 = px.scatter(df, x="umap_cluster", y="effective_rent_growth", color="umap_cluster", title="UMAP Cluster vs Rent Growth")
    st.plotly_chart(fig7, use_container_width=True)

    fig8 = px.histogram(df, x="property_age", title="Property Age Distribution")
    st.plotly_chart(fig8, use_container_width=True)

    fig9 = px.pie(df, names="large_project_flag", title="Large Project Flag Proportion")
    st.plotly_chart(fig9, use_container_width=True)

else:
    st.sidebar.warning("Please enter your OpenAI API key to use the assistant.")
