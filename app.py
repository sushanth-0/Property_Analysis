import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import openai
import numpy as np

try:
    import faiss
except ImportError:
    import faiss_cpu as faiss

client = openai.OpenAI()

@st.cache_resource
def build_faiss_index(csv_file: str):
    df = pd.read_csv(csv_file)
    summary = f"SUMMARY: Total rows = {df.shape[0]}"
    texts = [summary] + ["Row {}: ".format(idx) + " | ".join([f"{col}: {row[col]}" for col in df.columns]) for idx, row in df.iterrows()]
    embeddings = [client.embeddings.create(model="text-embedding-3-large", input=t).data[0].embedding for t in texts]
    matrix = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index, texts, df

def get_relevant_chunks(query, index, texts, top_k=10):
    embedding = client.embeddings.create(model="text-embedding-3-large", input=query).data[0].embedding
    query_vector = np.array(embedding, dtype=np.float32).reshape(1, -1)
    _, idxs = index.search(query_vector, top_k)
    return [texts[i] for i in idxs[0]]

def answer_query(query, index, texts):
    context = "\n\n".join(get_relevant_chunks(query, index, texts))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Use ONLY the given data."},
            {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return response.choices[0].message.content

st.set_page_config(layout="wide")
st.title("Lease-Up RAG Insights Dashboard")

components.html(f'''<iframe width="100%" height="100%" src="https://lookerstudio.google.com/embed/reporting/b3fcc2c4-24c5-4869-b128-c71e658b3f16/page/7m1DF" frameborder="0" style="border:0;height:calc(100vh - 4rem)" allowfullscreen></iframe>''', height=800)

with st.sidebar:
    st.markdown("## Ask Your Data")
    api_key = st.text_input("OpenAI API Key", type="password")
    user_q = st.text_input("", placeholder="Ask about lease-up data...")
    button = st.button("Generate Answer")

if api_key:
    client.api_key = api_key
    index, texts, df = build_faiss_index("Market1_Final_Selected.csv")
    if button and user_q:
        reply = answer_query(user_q, index, texts)
        st.sidebar.subheader("Answer")
        st.sidebar.write(reply)
else:
    st.sidebar.warning("Please enter your OpenAI API key to use the assistant.")
