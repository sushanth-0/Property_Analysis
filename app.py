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

    @st.cache_resource
    def build_index(api_key):
        temp_client = OpenAI(api_key=api_key)
        chunks = []
        embeddings = []
        for _, row in df.iterrows():
            text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
            chunks.append(text)
            emb = temp_client.embeddings.create(model="text-embedding-3-large", input=text).data[0].embedding
            embeddings.append(emb)
        vectors = np.array(embeddings, dtype=np.float32)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(vectors)
        index.add(vectors)
        return index, chunks

    index, chunks = build_index(api_key)

    if button and user_q:
        true_average = df["leaseup_time"].dropna().mean()

        q_emb = client.embeddings.create(model="text-embedding-3-large", input=user_q).data[0].embedding
        q_vec = np.array(q_emb, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        distances, ids = index.search(q_vec, len(df))
        retrieved = [chunks[i] for i in ids[0]]

        context = "\n\n".join(retrieved)
        prompt = (
            f"Rows:\n{context}\n\n"
            f"Note: The true lease-up time average for ALL {len(df)} rows is {true_average:.2f} months."
            f" Use this number if needed.\n\nQuestion: {user_q}"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a lease-up analyst. Use all rows and the true average."},
                {"role": "user", "content": prompt}
            ]
        )
        st.sidebar.subheader("Answer")
        st.sidebar.write(response.choices[0].message.content)
else:
    st.sidebar.warning("Enter your OpenAI API key.")

st.header(f"Market Dashboard: {market}")

# === Use one per row with side margins ===

def plot_centered(fig):
    left, center, right = st.columns([1, 6, 1])
    with center:
        st.plotly_chart(fig, use_container_width=True)

line_df = market_df.groupby("delivery_year").size().reset_index(name="Count")
fig1 = px.line(line_df, x="delivery_year", y="Count", title="Properties Delivered per Year")
plot_centered(fig1)

submarket_counts = market_df["Submarket"].value_counts().reset_index()
submarket_counts.columns = ["Submarket", "Count"]
fig2 = px.bar(submarket_counts, x="Submarket", y="Count", title="Properties by Submarket")
plot_centered(fig2)

fig3 = px.histogram(market_df, x="leaseup_time", nbins=30, title="Lease-Up Time Distribution")
plot_centered(fig3)

fig4 = px.scatter(market_df, x="effective_rent_delivery", y="effective_rent_leaseup",
                  color="Submarket", title="Delivery Rent vs Lease-Up Rent")
plot_centered(fig4)

fig5 = px.box(market_df, y="effective_rent_growth", title="Effective Rent Growth Boxplot")
plot_centered(fig5)

fig6 = px.pie(market_df, names="negative_growth", title="Negative Growth Proportion")
plot_centered(fig6)

fig7 = px.scatter(market_df, x="umap_cluster", y="effective_rent_growth", color="umap_cluster",
                  title="Clusters vs Rent Growth")
plot_centered(fig7)

fig8 = px.histogram(market_df, x="property_age", nbins=20, title="Property Age Distribution")
plot_centered(fig8)

fig9 = px.pie(market_df, names="large_project_flag", title="Large Project Flag Proportion")
plot_centered(fig9)
