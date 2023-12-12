import streamlit as st
from sklearn.tree import DecisionTreeClassifier


def dt_param_selector():

    criterion = st.selectbox("Criterion", ["gini", "entropy"])
    max_depth = st.number_input("Max Depth", 1, 50, 5, 1)
    min_samples_split = st.number_input("Min Sample Split", 1, 20, 2, 1)
    max_features = st.selectbox("Max Features", [None, "Auto", "Sqrt", "log2"])

    params = {
        "criterion": criterion,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "max_features": max_features,
    }

    model = DecisionTreeClassifier(**params)
    return model
