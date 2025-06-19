import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import openai

@st.cache_data
def load_data():
    df1 = pd.read_csv("Market1_Final_Selected.csv")
    df1["Market"] = "Market 1"
    df2 = pd.read_csv("Market2_Final_Selected.csv")
    df2["Market"] = "Market 2"
    return pd.concat([df1, df2], ignore_index=True)

df = load_data()

st.header("Ask the GenAI")
api_key = st.text_input("OpenAI API Key", type="password")
user_question = st.text_area("Your Question")

if st.button("Get Answer"):
    if api_key and user_question:
        openai.api_key = api_key
        context = f"Columns: delivery_year, Submarket, leaseup_time, effective_rent_delivery, effective_rent_leaseup, effective_rent_growth, negative_growth, umap_cluster, property_age, large_project_flag. Sample: {df.head(5).to_dict()}"
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful analyst assistant."},
                {"role": "user", "content": f"{context} Question: {user_question}"}
            ]
        )
        st.write(response.choices[0].message.content)
    else:
        st.warning("Enter API key and question.")

st.sidebar.title("Lease-Up Dashboard")
market = st.sidebar.selectbox("Market", df["Market"].unique())
submarkets = st.sidebar.multiselect("Submarkets", df[df["Market"] == market]["Submarket"].unique(), df[df["Market"] == market]["Submarket"].unique())
years = st.sidebar.slider("Delivery Year", int(df["delivery_year"].min()), int(df["delivery_year"].max()), (int(df["delivery_year"].min()), int(df["delivery_year"].max())))
filtered_df = df[(df["Market"] == market) & (df["Submarket"].isin(submarkets)) & (df["delivery_year"].between(*years))]

st.title("Property Lease-Up Dashboard")
st.write(f"Market: {market}")

features = ["delivery_year", "Submarket", "leaseup_time", "effective_rent_delivery", "effective_rent_leaseup", "effective_rent_growth", "negative_growth", "umap_cluster", "property_age", "large_project_flag"]

for feature in features:
    st.subheader(feature)
    if filtered_df[feature].dtype == 'object' or filtered_df[feature].dtype == 'bool':
        fig = px.bar(filtered_df, x=feature, title=f"{feature} Count")
    else:
        fig = go.Figure()
        fig.add_trace(go.Box(y=filtered_df[feature], name=feature))
        fig.update_layout(title=f"{feature} Box Plot")
    st.plotly_chart(fig, use_container_width=True)
