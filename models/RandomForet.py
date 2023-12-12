import streamlit as st
from sklearn.ensemble import RandomForestClassifier


def rf_param_selector():

    criterion = st.selectbox("Criterion", ["gini", "entropy"])
    n_estimators = st.number_input("N estimators", 50, 300, 100, 10)
    max_depth = st.number_input("Max Depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("Min Sample Split", 1, 20, 2, 1)
    max_features = st.selectbox("Max Features", [None, "auto", "sqrt", "log2"])

    params = {
        "criterion": criterion,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    return model
