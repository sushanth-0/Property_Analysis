import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Market1_Final_Selected.csv")
    df["Market"] = "Market 1"
    return df

df = load_data()

# Sidebar
st.sidebar.title("Lease-Up Dashboard with GenAI")
st.sidebar.write("Filter the data:")

submarkets = st.sidebar.multiselect("Submarkets", options=df["Submarket"].unique(), default=df["Submarket"].unique())
years = st.sidebar.slider("Delivery Year", int(df["delivery_year"].min()), int(df["delivery_year"].max()), (int(df["delivery_year"].min()), int(df["delivery_year"].max())))

filtered_df = df[(df["Submarket"].isin(submarkets)) & (df["delivery_year"].between(*years))]

# Main
st.title("Property Lease-Up Analysis with GenAI")
st.write("Explore lease-up times, rent growth, and clusters.")

# Charts
col1, col2 = st.columns(2)

with col1:
    fig1 = px.scatter(filtered_df, x="delivery_year", y="leaseup_time", color="Submarket",
                      title="Lease-Up Time vs Delivery Year", hover_data=["effective_rent_growth"])
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(filtered_df, x="effective_rent_growth", color="negative_growth",
                        nbins=30, title="Rent Growth Distribution")
    st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(filtered_df, x="effective_rent_delivery", y="effective_rent_leaseup",
                   color="umap_cluster", title="Clusters of Similar Properties",
                   hover_data=["Submarket", "delivery_year"])
st.plotly_chart(fig3, use_container_width=True)


st.header("Ask the GenAI about this data")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")
user_question = st.text_area("Ask a question about lease-up trends:")

if st.button("Get Answer"):
    if api_key and user_question:
        openai.api_key = api_key
        context = f"The user is asking about lease-up trends for properties in Market 1 with columns: delivery_year, Submarket, leaseup_time, effective_rent_delivery, effective_rent_leaseup, effective_rent_growth, negative_growth, umap_cluster, property_age, large_project_flag. Here are some rows: {filtered_df.head(5).to_dict()}"
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful analyst assistant."},
                {"role": "user", "content": f"{context} \n\nQuestion: {user_question}"}
            ]
        )
        st.write(response.choices[0].message.content)
    else:
        st.warning("Please enter both your OpenAI API key and a question.")
