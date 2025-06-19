import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from openai import OpenAI

st.set_page_config(layout="wide")
st.title("Lease-Up 10-Feature RAG Insights Dashboard")

components.html(f'''<iframe width="100%" height="100%" src="https://lookerstudio.google.com/embed/reporting/b3fcc2c4-24c5-4869-b128-c71e658b3f16/page/7m1DF" frameborder="0" style="border:0;height:calc(100vh - 4rem)" allowfullscreen></iframe>''', height=800)

with st.sidebar:
    st.markdown("## Ask Your Lease-Up Data (10 features)")
    api_key = st.text_input("OpenAI API Key", type="password")
    user_q = st.text_area("Your Question", placeholder="Ask anything about delivery year, submarket, lease-up time, rent, growth, clusters, project flags, etc.")
    button = st.button("Generate Answer")

if api_key:
    client = OpenAI(api_key=api_key)

    @st.cache_data
    def load_data(csv_file):
        df = pd.read_csv(csv_file)
        summary = f"SUMMARY: Total rows = {df.shape[0]}"
        texts = [summary] + [
            "Row {}: ".format(idx) + " | ".join([f"{col}: {row[col]}" for col in df.columns])
            for idx, row in df.iterrows()
        ]
        return texts, df

    def answer_query_rag(query, texts):
        context = "\n\n".join(texts)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a smart lease-up assistant. Use ONLY the data provided for the 10 key features. Do not make up extra facts."},
                {"role": "user", "content": f"Data:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content

    texts, df = load_data("Market1_Final_Selected.csv")
    if button and user_q:
        reply = answer_query_rag(user_q, texts)
        st.sidebar.subheader("Answer")
        st.sidebar.write(reply)
else:
    st.sidebar.warning("Please enter your OpenAI API key to use the assistant.")
