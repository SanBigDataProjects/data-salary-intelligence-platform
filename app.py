import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Data Salary Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# LOAD & CLEAN DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Data/ds_salaries.csv")

    # Focus on core roles
    core_roles = ["Data Scientist", "Data Engineer", "Data Analyst"]
    df = df[df["job_title"].isin(core_roles)]

    # Full-time only
    df = df[df["employment_type"] == "FT"]

    # Remove extreme low outliers
    df = df[df["salary_in_usd"] >= 20000]

    # Simplify location
    df["company_location"] = df["company_location"].apply(
        lambda x: x if x in ["US", "GB", "CA"] else "Other"
    )

    # Create remote_type
    df["remote_type"] = df["remote_ratio"].map({
        0: "Onsite",
        50: "Hybrid",
        100: "Remote"
    })

    # Drop unused columns
    df = df.drop(columns=[
        "Unnamed: 0",
        "salary",
        "salary_currency",
        "employment_type",
        "employee_residence",
        "remote_ratio"
    ])

    return df

df = load_data()

# ---------------------------------------------------
# FEATURE & TARGET SPLIT
# ---------------------------------------------------
categorical_cols = [
    "experience_level",
    "job_title",
    "company_location",
    "company_size",
    "remote_type"
]

numeric_cols = ["work_year"]

X = df.drop(columns=["salary_in_usd"])
y = df["salary_in_usd"]

# ---------------------------------------------------
# PREPROCESSOR
# ---------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

# ---------------------------------------------------
# MODELS
# ---------------------------------------------------
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
])

lr_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

rf_r2 = r2_score(y_test, rf_model.predict(X_test))
rf_mae = mean_absolute_error(y_test, rf_model.predict(X_test))

lr_r2 = r2_score(y_test, lr_model.predict(X_test))
lr_mae = mean_absolute_error(y_test, lr_model.predict(X_test))

# ---------------------------------------------------
# APP HEADER
# ---------------------------------------------------
st.title("📊 Data Salary Intelligence Platform")

st.markdown("""
This portfolio project analyzes compensation trends across core data roles 
(Data Scientist, Data Engineer, Data Analyst) and builds an interpretable 
machine learning model to predict salary based on structured job attributes.
""")

st.divider()

# ---------------------------------------------------
# TABS
# ---------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📌 Project Overview",
     "📊 Data & Methodology",
     "📈 Market Insights",
     "🔍 Model Interpretability",
     "💰 Salary Predictor"]
)

# ---------------------------------------------------
# TAB 1 - PROJECT OVERVIEW
# ---------------------------------------------------
with tab1:
    st.subheader("🎯 Project Objective")

    st.markdown("""
The objective of this project is to identify and quantify the primary drivers 
of salary variation in data-related roles and develop a predictive model 
to estimate compensation.

This project aims to:

• Evaluate the impact of experience level  
• Measure geographic compensation differences  
• Analyze remote vs onsite salary premiums  
• Compare compensation across roles  
• Build an interpretable machine learning model  
""")

    st.subheader("📊 Model Performance Comparison")

    performance_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R² Score": [lr_r2, rf_r2],
        "MAE (USD)": [int(lr_mae), int(rf_mae)]
    })

    st.dataframe(performance_df, use_container_width=True)

    st.markdown("""
Random Forest improves performance by capturing nonlinear interactions 
between experience, geography, and role-based compensation dynamics.
""")

# ---------------------------------------------------
# TAB 2 - DATA & METHODOLOGY
# ---------------------------------------------------
with tab2:
    st.subheader("Dataset Summary")
    st.write(f"Total Records Used: {len(df)}")
    st.write("Roles Included: Data Scientist, Data Engineer, Data Analyst")

    st.subheader("Modeling Approach")

    st.markdown("""
1. Data Cleaning & Filtering  
2. Feature Engineering (Remote Type, Location Simplification)  
3. One-Hot Encoding via ColumnTransformer  
4. Baseline Linear Regression  
5. Random Forest Ensemble Model  
6. Train/Test Evaluation  
""")

# ---------------------------------------------------
# TAB 3 - MARKET INSIGHTS
# ---------------------------------------------------
with tab3:
    st.subheader("📊 Market Snapshot")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Salary",
        f"${int(df['salary_in_usd'].mean()):,}"
    )

    col2.metric(
        "Highest Paying Role",
        df.groupby("job_title")["salary_in_usd"].mean().idxmax()
    )

    col3.metric(
        "Remote Roles %",
        f"{int((df['remote_type']=='Remote').mean()*100)}%"
    )

    st.divider()

    st.subheader("Average Salary by Role")
    fig_role = px.bar(
        df.groupby("job_title")["salary_in_usd"].mean().reset_index(),
        x="job_title",
        y="salary_in_usd",
        color="job_title",
        title="Salary by Role"
    )
    st.plotly_chart(fig_role, use_container_width=True)

    st.subheader("Salary by Experience Level")
    fig_exp = px.bar(
        df.groupby("experience_level")["salary_in_usd"].mean().reset_index(),
        x="experience_level",
        y="salary_in_usd",
        color="experience_level"
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    st.subheader("Salary by Work Type")
    fig_remote = px.bar(
        df.groupby("remote_type")["salary_in_usd"].mean().reset_index(),
        x="remote_type",
        y="salary_in_usd",
        color="remote_type"
    )
    st.plotly_chart(fig_remote, use_container_width=True)

# ---------------------------------------------------
# TAB 4 - FEATURE IMPORTANCE
# ---------------------------------------------------
with tab4:
    st.subheader("Top Salary Drivers (Random Forest)")

    feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
    importances = rf_model.named_steps["regressor"].feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig_importance = px.bar(
        feature_importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig_importance.update_layout(yaxis=dict(autorange="reversed"))

    st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("""
Geographic location (particularly US-based roles), experience level, 
and job title are the strongest compensation drivers in the model.
""")

# ---------------------------------------------------
# TAB 5 - SALARY PREDICTOR
# ---------------------------------------------------
with tab5:
    st.subheader("Predict Salary")

    role = st.selectbox("Job Role", df["job_title"].unique())
    experience = st.selectbox("Experience Level", df["experience_level"].unique())
    location = st.selectbox("Company Location", df["company_location"].unique())
    company_size = st.selectbox("Company Size", df["company_size"].unique())
    remote_type = st.selectbox("Work Type", df["remote_type"].unique())
    year = st.selectbox("Work Year", sorted(df["work_year"].unique()))

    input_data = pd.DataFrame({
        "experience_level": [experience],
        "job_title": [role],
        "company_location": [location],
        "company_size": [company_size],
        "remote_type": [remote_type],
        "work_year": [year]
    })

    if st.button("Predict Salary"):
        prediction = rf_model.predict(input_data)[0]
        st.success(f"Estimated Salary: ${int(prediction):,}")


