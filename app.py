import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="California Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ======================================
# LOAD CSS
# ======================================

def load_css(file_name):

    with open(file_name) as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css("style.css")

# ======================================
# LOAD DATA
# ======================================

@st.cache_data
def load_data():

    housing = fetch_california_housing()

    df = pd.DataFrame(
        housing.data,
        columns=housing.feature_names
    )

    df["Price"] = housing.target

    return df

df = load_data()

# ======================================
# FEATURES & TARGET
# ======================================

X = df.drop("Price", axis=1)

y = df["Price"]

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================
# FEATURE SCALING
# ======================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

# ======================================
# MODEL TRAINING
# ======================================

model = SVR(kernel="rbf")

model.fit(X_train, y_train)

# ======================================
# PREDICTIONS
# ======================================

y_pred = model.predict(X_test)

# ======================================
# METRICS
# ======================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

# ======================================
# HEADER
# ======================================

st.markdown("""
<div class="main-header">

<h1>🏠 California Housing Price Prediction</h1>

<p>
Machine Learning Dashboard using SVM Regression
</p>

</div>
""", unsafe_allow_html=True)

# ======================================
# SIDEBAR
# ======================================

st.sidebar.markdown("""
<div class="sidebar-title">
🏠 Housing ML App
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

section = st.sidebar.radio(
    "📌 Navigation Menu",
    [
        "🏠 Dashboard",
        "📋 Dataset Overview",
        "📊 Visualizations",
        "📈 Evaluation",
        "🤖 Prediction"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info("""
This project predicts California Housing Prices using:

✅ Support Vector Regression (SVR)

✅ Scikit-Learn

✅ Streamlit Dashboard
""")

# ======================================
# DASHBOARD
# ======================================

if section == "🏠 Dashboard":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📊 Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Average Price",
        f"{df['Price'].mean():.2f}"
    )

    c4.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # ==================================
    # HEATMAP
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        df.corr(),
        annot=True,
        cmap="Blues",
        linewidths=1
    )

    st.pyplot(fig)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ======================================
# DATASET OVERVIEW
# ======================================

elif section == "📋 Dataset Overview":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📋 Dataset Preview")

    st.dataframe(df.head(10))

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # ==================================
    # DATA INFO
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🧾 Dataset Information")

    info_df = pd.DataFrame({

        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values

    })

    st.table(info_df)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # ==================================
    # STATISTICS
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📈 Statistical Summary")

    st.dataframe(df.describe())

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ======================================
# VISUALIZATIONS
# ======================================

elif section == "📊 Visualizations":

    # ==================================
    # MEDIAN INCOME VS PRICE
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("💰 Median Income vs House Price")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.scatterplot(
        x=df["MedInc"],
        y=df["Price"],
        color="skyblue"
    )

    st.pyplot(fig)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # ==================================
    # HOUSE AGE DISTRIBUTION
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("🏡 House Age Distribution")

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        df["HouseAge"],
        kde=True,
        color="gray"
    )

    st.pyplot(fig)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    # ==================================
    # ACTUAL VS PREDICTED
    # ==================================

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📉 Actual vs Predicted Prices")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        y_test,
        y_pred,
        color="skyblue"
    )

    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")

    st.pyplot(fig)

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ======================================
# EVALUATION
# ======================================

elif section == "📈 Evaluation":

    st.markdown(
        '<div class="card">',
        unsafe_allow_html=True
    )

    st.subheader("📊 Model Evaluation")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "MAE",
        f"{mae:.2f}"
    )

    c2.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    c3.metric(
        "R² Score",
        f"{r2:.2f}"
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

# ======================================
# PREDICTION
# ======================================

elif section == "🤖 Prediction":

    st.markdown("""
    <div class="card">

    <h2 style='text-align:center;'>
    🤖 California House Price Prediction
    </h2>

    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ==================================
    # LEFT COLUMN
    # ==================================

    with col1:

        medinc = st.slider(
            "Median Income",
            float(df["MedInc"].min()),
            float(df["MedInc"].max()),
            3.0
        )

        houseage = st.slider(
            "House Age",
            float(df["HouseAge"].min()),
            float(df["HouseAge"].max()),
            20.0
        )

        averooms = st.slider(
            "Average Rooms",
            float(df["AveRooms"].min()),
            float(df["AveRooms"].max()),
            5.0
        )

        avebedrms = st.slider(
            "Average Bedrooms",
            float(df["AveBedrms"].min()),
            float(df["AveBedrms"].max()),
            1.0
        )

    # ==================================
    # RIGHT COLUMN
    # ==================================

    with col2:

        population = st.slider(
            "Population",
            float(df["Population"].min()),
            float(df["Population"].max()),
            1000.0
        )

        aveoccup = st.slider(
            "Average Occupancy",
            float(df["AveOccup"].min()),
            float(df["AveOccup"].max()),
            3.0
        )

        latitude = st.slider(
            "Latitude",
            float(df["Latitude"].min()),
            float(df["Latitude"].max()),
            34.0
        )

        longitude = st.slider(
            "Longitude",
            float(df["Longitude"].min()),
            float(df["Longitude"].max()),
            -118.0
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ==================================
    # PREDICTION BUTTON
    # ==================================

    if st.button("🚀 Predict House Price"):

        input_data = np.array([[
            medinc,
            houseage,
            averooms,
            avebedrms,
            population,
            aveoccup,
            latitude,
            longitude
        ]])

        input_data = scaler.transform(
            input_data
        )

        prediction = model.predict(
            input_data
        )[0]

        st.markdown(
            f"""
            <div class="prediction-box">

            <h2>🏡 Predicted House Price</h2>

            <h1>💲 {prediction:.2f}</h1>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.success(
            "Prediction Generated Successfully ✅"
        )