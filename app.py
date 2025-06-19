import streamlit as st
import pandas as pd
from openai import OpenAI

@st.cache_data
def load_data():
    df1 = pd.read_csv("Market1_Final_Selected.csv")
    df1["Market"] = "Market 1"
    df2 = pd.read_csv("Market2_Final_Selected.csv")
    df2["Market"] = "Market 2"
    return pd.concat([df1, df2], ignore_index=True)

df = load_data()

st.title("GenAI Lease-Up Assistant")

market = st.selectbox("Select Market", df["Market"].unique())

# Let user select additional feature to include in context
features = [
    "delivery_year", "Submarket", "leaseup_time", "effective_rent_delivery",
    "effective_rent_leaseup", "effective_rent_growth", "negative_growth",
    "umap_cluster", "property_age", "large_project_flag"
]

selected_features = st.multiselect("Select features to include in the GenAI context", features, default=features)

api_key = st.text_input("OpenAI API Key", type="password")
user_question = st.text_area("Ask your question about lease-up data")

if st.button("Get Answer"):
    if api_key and user_question:
        try:
            client = OpenAI(api_key=api_key)
            market_df = df[df["Market"] == market][selected_features]
            avg_leaseup_time_market = market_df["leaseup_time"].dropna().mean() if "leaseup_time" in selected_features else "N/A"
            context = f"Selected Market: {market}; Total rows: {market_df.shape[0]}; Features included: {selected_features}; Average lease-up time: {avg_leaseup_time_market}" + "\nSample data:" + str(market_df.head(10).to_dict())
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful lease-up data assistant."},
                    {"role": "user", "content": f"{context} | Question: {user_question}"}
                ]
            )
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter your API key and a question.")
