import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import plotly.express as px


# =======================================================
# PAGE CONFIGURATION
# =======================================================
st.set_page_config(
    page_title="FINAL PROJECT - IRIS SPECIES CLASSIFICATION",
    layout="wide"
)

st.title("FINAL PROJECT – IRIS SPECIES CLASSIFICATION")

st.markdown("""
This project builds a machine learning model to classify Iris flower species 
based on four numerical measurements.  
The application includes dataset exploration, model training, performance evaluation, 
and real-time predictions using user-provided values.
""")

st.markdown("---")
st.markdown("**Developed by:** Elis García · Neris Pacheco")
st.markdown("---")


# =======================================================
# DATA LOADING FUNCTION
# =======================================================
@st.cache_data
def load_dataset():
    iris_raw = load_iris(as_frame=True)
    df = iris_raw.frame.copy()
    df.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Target"]
    df["Species"] = df["Target"].map({i: name for i, name in enumerate(iris_raw.target_names)})
    return df, iris_raw

dataset, iris_info = load_dataset()


# =======================================================
# SIDEBAR CONFIGURATION PANEL
# =======================================================
st.sidebar.header("Model Parameters")

test_fraction = st.sidebar.select_slider(
    "Test Set Fraction",
    options=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
    value=0.25
)

random_seed = st.sidebar.number_input("Random Seed", min_value=0, value=42)

num_trees = st.sidebar.slider("Number of Trees", 20, 350, 150, 10)
max_depth = st.sidebar.slider("Maximum Depth (0 = Unlimited)", 0, 40, 0)

st.sidebar.markdown("---")
st.sidebar.header("User Input for Prediction")

u_sl = st.sidebar.number_input("Sepal Length", min_value=0.0, value=5.0)
u_sw = st.sidebar.number_input("Sepal Width", min_value=0.0, value=3.2)
u_pl = st.sidebar.number_input("Petal Length", min_value=0.0, value=4.3)
u_pw = st.sidebar.number_input("Petal Width", min_value=0.0, value=1.3)

user_sample = np.array([[u_sl, u_sw, u_pl, u_pw]])


# =======================================================
# PREPROCESSING
# =======================================================
X = dataset[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
y = dataset["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_fraction, random_state=random_seed, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
user_scaled = scaler.transform(user_sample)


# =======================================================
# MODEL TRAINING
# =======================================================
rf = RandomForestClassifier(
    n_estimators=num_trees,
    max_depth=None if max_depth == 0 else max_depth,
    random_state=random_seed
)

rf.fit(X_train_scaled, y_train)
test_predictions = rf.predict(X_test_scaled)


# =======================================================
# PERFORMANCE METRICS
# =======================================================
colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("Model Performance")

    st.write(f"**Accuracy:** {accuracy_score(y_test, test_predictions):.3f}")
    st.write(f"**Precision (weighted):** {precision_score(y_test, test_predictions, average='weighted'):.3f}")
    st.write(f"**Recall (weighted):** {recall_score(y_test, test_predictions, average='weighted'):.3f}")
    st.write(f"**F1-score (weighted):** {f1_score(y_test, test_predictions, average='weighted'):.3f}")

    st.markdown("### Classification Report")
    st.text(classification_report(y_test, test_predictions, target_names=iris_info.target_names))

    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, test_predictions)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels={"x": "Predicted", "y": "True"},
        x=iris_info.target_names,
        y=iris_info.target_names
    )
    st.plotly_chart(fig_cm, use_container_width=True)


with colB:
    st.subheader("Dataset Preview")
    st.dataframe(dataset.head())

    st.subheader("Scatter Matrix")
    matrix_fig = px.scatter_matrix(
        dataset,
        dimensions=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"],
        color="Species"
    )
    st.plotly_chart(matrix_fig, use_container_width=True)


# =======================================================
# USER PREDICTION SECTION
# =======================================================
st.subheader("Prediction for User Input")

predicted_label = rf.predict(user_scaled)[0]
prediction_probabilities = rf.predict_proba(user_scaled)[0]

st.write(f"**Predicted Species:** {iris_info.target_names[predicted_label]}")

prob_df = pd.DataFrame({
    "Species": iris_info.target_names,
    "Probability": prediction_probabilities
})

st.dataframe(prob_df)


# =======================================================
# 3D SCATTER PLOT WITH USER SAMPLE
# =======================================================
visual_data = dataset.copy()
visual_data["PointType"] = "Dataset"

user_row = {
    "SepalLength": u_sl,
    "SepalWidth": u_sw,
    "PetalLength": u_pl,
    "PetalWidth": u_pw,
    "Species": iris_info.target_names[predicted_label],
    "PointType": "User"
}

visual_data = pd.concat([visual_data, pd.DataFrame([user_row])], ignore_index=True)

fig3d = px.scatter_3d(
    visual_data,
    x="PetalLength",
    y="PetalWidth",
    z="SepalLength",
    color="Species",
    symbol="PointType",
    size=[12 if pt == "User" else 6 for pt in visual_data["PointType"]]
)

fig3d.update_layout(height=550)

st.subheader("3D Visualization: User Sample Within the Dataset")
st.plotly_chart(fig3d, use_container_width=True)


st.markdown("---")
st.markdown("**Developed by:** Elis García · Neris Pacheco")
