import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import plotly.express as px


# =========================================================
# PAGE HEADER
# =========================================================
st.set_page_config(
    page_title="FINAL PROJECT - IRIS SPECIES CLASSIFICATION",
    layout="wide"
)

st.title("FINAL PROJECT – IRIS SPECIES CLASSIFICATION")
st.write("""
This dashboard presents a complete machine learning workflow for classifying Iris flower species.
It includes model performance metrics, an interactive prediction panel, and visual tools to explore the dataset.
""")

st.markdown("---")
st.markdown("**Developed by:** Elis García · Neris Pacheco")
st.markdown("---")


# =========================================================
# DATA LOADING
# =========================================================
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Target"]
    df["Species"] = df["Target"].map({i: name for i, name in enumerate(iris.target_names)})
    return df, iris

df, iris_info = load_data()


# =========================================================
# MODEL TRAINING (fixed to avoid user breaking logic)
# =========================================================
X = df[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=12, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=180, max_depth=None, random_state=12)
model.fit(X_train_scaled, y_train)
pred_test = model.predict(X_test_scaled)


# =========================================================
# SECTION 1 — MODEL METRICS
# =========================================================
st.header("1. Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, pred_test):.3f}")
col2.metric("Precision", f"{precision_score(y_test, pred_test, average='weighted'):.3f}")
col3.metric("Recall", f"{recall_score(y_test, pred_test, average='weighted'):.3f}")
col4.metric("F1 Score", f"{f1_score(y_test, pred_test, average='weighted'):.3f}")

st.subheader("Classification Report")
st.text(classification_report(y_test, pred_test, target_names=iris_info.target_names))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, pred_test)
fig_cm = px.imshow(
    cm,
    text_auto=True,
    labels={"x": "Predicted", "y": "Actual"},
    x=iris_info.target_names,
    y=iris_info.target_names
)
st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")


# =========================================================
# SECTION 2 — USER INPUT + PREDICTION + 3D PLOT
# =========================================================
st.header("2. Interactive Prediction Panel")

st.write("""
Enter sepal and petal measurements below.  
The model will classify the flower and display its position relative to the dataset in a 3D plot.
""")

colA, colB = st.columns(2)

with colA:
    st.subheader("Enter Flower Measurements")

    u_sl = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
    u_sw = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
    u_pl = st.number_input("Petal Length (cm)", min_value=0.0, value=1.4)
    u_pw = st.number_input("Petal Width (cm)", min_value=0.0, value=0.3)

    user_sample = np.array([[u_sl, u_sw, u_pl, u_pw]])
    user_scaled = scaler.transform(user_sample)

    pred_label = model.predict(user_scaled)[0]
    pred_probs = model.predict_proba(user_scaled)[0]
    species_name = iris_info.target_names[pred_label]

    st.subheader("Prediction Result")
    st.write(f"**Predicted Species:** {species_name}")

    st.write("Prediction Probabilities:")
    st.dataframe(pd.DataFrame({
        "Species": iris_info.target_names,
        "Probability": pred_probs
    }))

with colB:
    st.subheader("3D Visualization of Prediction")

    plot_df = df.copy()
    plot_df["PointType"] = "Dataset"

    new_point = {
        "SepalLength": u_sl,
        "SepalWidth": u_sw,
        "PetalLength": u_pl,
        "PetalWidth": u_pw,
        "Species": species_name,
        "PointType": "User"
    }

    plot_df = pd.concat([plot_df, pd.DataFrame([new_point])], ignore_index=True)

    fig3d = px.scatter_3d(
        plot_df,
        x="PetalLength",
        y="PetalWidth",
        z="SepalLength",
        color="Species",
        symbol="PointType",
        size=[12 if t == "User" else 5 for t in plot_df["PointType"]]
    )

    fig3d.update_layout(height=550)
    st.plotly_chart(fig3d, use_container_width=True)

st.markdown("---")


# =========================================================
# SECTION 3 — ADDITIONAL DATA VISUALIZATIONS
# =========================================================
st.header("3. Dataset Visual Exploration")

st.subheader("Histogram Distributions")
hist_cols = st.multiselect(
    "Select features to visualize:",
    ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"],
    default=["SepalLength", "PetalLength"]
)

if hist_cols:
    for col in hist_cols:
        st.write(f"**Histogram of {col}**")
        fig_hist = px.histogram(df, x=col, color="Species", nbins=20)
        st.plotly_chart(fig_hist, use_container_width=True)

st.subheader("Scatter Matrix")
scatter_fig = px.scatter_matrix(
    df,
    dimensions=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"],
    color="Species"
)
st.plotly_chart(scatter_fig, use_container_width=True)

st.markdown("---")
st.markdown("**Developed by: Elis García · Neris Pacheco**")
