import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import openai
import faiss
import numpy as np

@st.cache_resource
def build_faiss_index(csv_file: str):
    df = pd.read_csv(csv_file)
    summary_chunk = f"SUMMARY: Total rows in the CSV = {df.shape[0]}."
    text_chunks = []
    for idx, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        full_text = f"Row {idx}: {row_text}"
        text_chunks.append(full_text)
    texts = [summary_chunk] + text_chunks
    embeddings = []
    for text in texts:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response["data"][0]["embedding"]
        embeddings.append(embedding)
    embedding_matrix = np.array(embeddings, dtype=np.float32)
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)
    return index, texts, df

def get_relevant_chunks(query: str, index, texts, top_k=10):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=query
    )
    query_embedding = np.array(response["data"][0]["embedding"], dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    relevant_texts = [texts[i] for i in indices[0]]
    return relevant_texts

def answer_query(query: str, index, texts):
    relevant_chunks = get_relevant_chunks(query, index, texts, top_k=10)
    combined_context = "\n\n".join(relevant_chunks)
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer ONLY using the provided data context."},
        {"role": "user", "content": f"Data:\n{combined_context}\n\nQuestion: {query}\n\nAnswer using only the data above."}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0
    )
    return response["choices"][0]["message"]["content"]

st.set_page_config(layout="wide")
st.title("Lease-Up RAG Insights")

# Embed the existing dashboard (replace with your Looker Studio URL if needed)
dashboard_url = "https://lookerstudio.google.com/embed/reporting/b3fcc2c4-24c5-4869-b128-c71e658b3f16/page/7m1DF"
iframe = f'''<iframe width="100%" height="100%" src="{dashboard_url}" frameborder="0" style="border:0; margin:0; padding:0; height: calc(100vh - 4rem);" allowfullscreen sandbox="allow-storage-access-by-user-activation allow-scripts allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>'''
components.html(iframe, height=800)

with st.sidebar:
    st.markdown("<h2 style='border-bottom: 1px solid #ccc; color: #3949ab;'>Ask Your Lease-Up Data</h2>", unsafe_allow_html=True)
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
    user_message = st.text_input("", placeholder="Ask a question from the lease-up data...")
    send_button = st.button('Generate Answer')

if openai_api_key:
    openai.api_key = openai_api_key
    index, texts, df = build_faiss_index("Market1_Final_Selected.csv")
    if send_button and user_message:
        query_lower = user_message.lower()
        if "total rows" in query_lower or "how many rows" in query_lower:
            answer = f"The CSV has {df.shape[0]} rows."
        elif "distribution of" in query_lower:
            answer = answer_query(user_message, index, texts)
        else:
            answer = answer_query(user_message, index, texts)
        st.sidebar.subheader("Answer")
        st.sidebar.write(answer)
else:
    st.sidebar.warning("Please enter your OpenAI API key to proceed.")
