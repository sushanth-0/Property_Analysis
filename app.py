import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

@st.cache_data
def load_data():
    df1 = pd.read_csv("Market1_Final_Selected.csv")
    df1["Market"] = "Market 1"
    df2 = pd.read_csv("Market2_Final_Selected.csv")
    df2["Market"] = "Market 2"
    return pd.concat([df1, df2], ignore_index=True)

df = load_data()

st.title("Property Analysis")

st.header("Ask your question below to get instant insights")

target_market = st.selectbox("Select Market", df["Market"].unique())

market_df = df[df["Market"] == target_market]
filtered_df = market_df.copy()

# For demonstration, show row count & avg leaseup
avg_leaseup_time_market = market_df["leaseup_time"].dropna().mean()

api_key = st.text_input("OpenAI API Key", type="password")
user_question = st.text_area("Ask your question")

if st.button("Get Answer"):
    if api_key and user_question:
        try:
            client = OpenAI(api_key=api_key)
            
            # ✅ Instead of only leaseup_time, pass all columns context:
            context_rows = []
            for idx, row in market_df.head(20).iterrows():  # Limit to 20 rows for prompt length
                row_str = " | ".join([f"{col}: {row[col]}" for col in market_df.columns])
                context_rows.append(f"Row {idx}: {row_str}")
            context = "\n\n".join(context_rows)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a smart assistant for property lease-up data. "
                            "Answer using ONLY the provided context. "
                            "Cover delivery year, submarket, leaseup_time, rents, growth, cluster, age, flags etc."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Data:\n{context}\n\nQuestion: {user_question}\n\nAnswer using only this data."
                    }
                ]
            )
            st.write(response.choices[0].message.content)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter your API key and a question.")

# === Visuals ===

line_df = filtered_df.groupby("delivery_year").size().reset_index(name="Count")
fig1 = px.line(line_df, x="delivery_year", y="Count", title="Properties Delivered per Year")
st.plotly_chart(fig1, use_container_width=True)

submarket_counts = filtered_df["Submarket"].value_counts().reset_index()
submarket_counts.columns = ["Submarket", "Count"]
fig2 = px.bar(submarket_counts, x="Submarket", y="Count", title="Properties by Submarket")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.histogram(filtered_df, x="leaseup_time", nbins=30, title="Lease-Up Time Distribution")
st.plotly_chart(fig3, use_container_width=True)

fig4 = px.scatter(filtered_df, x="effective_rent_delivery", y="effective_rent_leaseup", color="Submarket", title="Delivery Rent vs Lease-Up Rent")
st.plotly_chart(fig4, use_container_width=True)

fig5 = px.box(filtered_df, y="effective_rent_growth", title="Effective Rent Growth Boxplot")
st.plotly_chart(fig5, use_container_width=True)

fig6 = px.pie(filtered_df, names="negative_growth", title="Negative Growth Proportion")
st.plotly_chart(fig6, use_container_width=True)

fig7 = px.scatter(filtered_df, x="umap_cluster", y="effective_rent_growth", color="umap_cluster", title="Clusters vs Rent Growth")
st.plotly_chart(fig7, use_container_width=True)

fig8 = px.histogram(filtered_df, x="property_age", nbins=20, title="Property Age Distribution")
st.plotly_chart(fig8, use_container_width=True)

fig9 = px.pie(filtered_df, names="large_project_flag", title="Large Project Flag Distribution")
st.plotly_chart(fig9, use_container_width=True)
