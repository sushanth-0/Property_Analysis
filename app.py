import streamlit as st
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
import plotly.express as px

st.set_page_config(layout="wide")
st.title("Dynamic RAG Lease-Up Assistant")

@st.cache_data
def load_data():
    df1 = pd.read_csv("Market1_Final_Selected.csv")
    df1["Market"] = "Market 1"
    df2 = pd.read_csv("Market2_Final_Selected.csv")
    df2["Market"] = "Market 2"
    return pd.concat([df1, df2], ignore_index=True)

df = load_data()

market = st.selectbox("Select Market", df["Market"].unique())
market_df = df[df["Market"] == market].reset_index(drop=True)

api_key = st.sidebar.text_input("OpenAI API Key", type="password")
user_q = st.sidebar.text_area("Question:")
button = st.sidebar.button("Generate Answer")

if api_key:
    client = OpenAI(api_key=api_key)

    if button and user_q:
        # Create embeddings on the fly only when question is asked
        chunks = []
        embeddings = []
        for _, row in market_df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in market_df.columns])
            chunks.append(text)
            emb = client.embeddings.create(model="text-embedding-3-large", input=text).data[0].embedding
            embeddings.append(emb)
        vectors = np.array(embeddings, dtype=np.float32)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        q_emb = client.embeddings.create(model="text-embedding-3-large", input=user_q).data[0].embedding
        q_vec = np.array(q_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        distances, ids = index.search(q_vec, 10)
        retrieved = [chunks[i] for i in ids[0]]

        context = "\n\n".join(retrieved)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a lease-up analyst. Use only the retrieved rows."},
                {"role": "user", "content": f"Rows:\n{context}\n\nQuestion: {user_q}"}
            ]
        )
        st.sidebar.subheader("Answer")
        st.sidebar.write(response.choices[0].message.content)
else:
    st.sidebar.warning("Enter your OpenAI API key.")

st.header(f"Market Dashboard: {market}")

line_df = market_df.groupby("delivery_year").size().reset_index(name="Count")
st.plotly_chart(px.line(line_df, x="delivery_year", y="Count"), use_container_width=True)

submarket_counts = market_df["Submarket"].value_counts().reset_index()
submarket_counts.columns = ["Submarket", "Count"]
st.plotly_chart(px.bar(submarket_counts, x="Submarket", y="Count"), use_container_width=True)

st.plotly_chart(px.histogram(market_df, x="leaseup_time", nbins=30), use_container_width=True)

st.plotly_chart(px.scatter(market_df, x="effective_rent_delivery", y="effective_rent_leaseup", color="Submarket"), use_container_width=True)

st.plotly_chart(px.box(market_df, y="effective_rent_growth"), use_container_width=True)

st.plotly_chart(px.pie(market_df, names="negative_growth"), use_container_width=True)

st.plotly_chart(px.scatter(market_df, x="umap_cluster", y="effective_rent_growth", color="umap_cluster"), use_container_width=True)

st.plotly_chart(px.histogram(market_df, x="property_age", nbins=20), use_container_width=True)

st.plotly_chart(px.pie(market_df, names="large_project_flag"), use_container_width=True)
