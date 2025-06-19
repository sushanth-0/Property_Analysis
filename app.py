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

# delivery_year: line chart
line_df = filtered_df.groupby("delivery_year").size().reset_index(name="count")
fig1 = px.line(line_df, x="delivery_year", y="count", title="Properties Delivered per Year")
st.plotly_chart(fig1, use_container_width=True)

# Submarket: fix to use proper columns
submarket_counts = filtered_df["Submarket"].value_counts().reset_index()
submarket_counts.columns = ["Submarket", "Count"]
fig2 = px.bar(submarket_counts, x="Submarket", y="Count", title="Properties by Submarket")
st.plotly_chart(fig2, use_container_width=True)

# Remaining plots stay the same
fig3 = px.histogram(filtered_df, x="leaseup_time", nbins=30, title="Lease-Up Time Distribution")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.scatter(filtered_df, x="effective_rent_delivery", y="effective_rent_leaseup", color="Submarket", title="Delivery Rent vs Lease-Up Rent")
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.box(filtered_df, y="effective_rent_growth", title="Effective Rent Growth Boxplot")
st.plotly_chart(fig5, use_container_width=True)

fig6 = px.pie(filtered_df, names="negative_growth", title="Negative Growth Distribution")
st.plotly_chart(fig6, use_container_width=True)

fig7 = px.scatter(filtered_df, x="umap_cluster", y="effective_rent_growth", color="umap_cluster", title="Clusters vs Rent Growth")
st.plotly_chart(fig7, use_container_width=True)

fig8 = px.histogram(filtered_df, x="property_age", nbins=20, title="Property Age Distribution")
st.plotly_chart(fig8, use_container_width=True)

fig9 = px.pie(filtered_df, names="large_project_flag", title="Large Project Flag Distribution")
st.plotly_chart(fig9, use_container_width=True)
