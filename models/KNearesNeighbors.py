import streamlit as st
from sklearn.neighbors import KNeighborsClassifier


def knn_param_selector():

    n_neighbors = st.number_input("n Neighbors", 5, 20, 5, 1)
    metric = st.selectbox(
        "Metric", ("minkowski", "euclidean", "manhattan", "chebyshev", "mahalanobis")
    )

    params = {"n_neighbors": n_neighbors, "metric": metric}

    model = KNeighborsClassifier(**params)
    return model
