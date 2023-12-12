import numpy as np
import streamlit as st


from models.NaiveBayes import nb_param_selector
from models.NeuralNetwork import nn_param_selector
from models.RandomForet import rf_param_selector
from models.DecisionTree import dt_param_selector
from models.LogisticRegression import lr_param_selector
from models.KNearesNeighbors import knn_param_selector
from models.SVC import svc_param_selector
from models.GradientBoosting import gb_param_selector

from models.utils import model_imports
from utils.functions import img_to_bytes


def introduction():
    st.title("**Machine Learning Playground**")
    st.markdown("""
                <style>
                .big-font {
                font-size:20px !important;
                    }
                </style>
                """, unsafe_allow_html=True)

    st.markdown('<p class="big-font"> Have you ever wondered how ML Models can learn from data and make predictions? My app here takes you on a visual tour of this process. I have created this project to show how different Machine Learning models learn and make decisions, and behave under noisy data, all in real-time. </p>', unsafe_allow_html=True)
    st.markdown('<p class="big-font"> You can choose from a variety of Machine Learning Models, tweak their hyper-parameters, change the noisiness level of the data and ultimately see how all of it affects the accuracy of the model to make predictions. </p>', unsafe_allow_html=True)

    st.markdown(
        """
    - Choose a dataset
    - Pick a model and set its hyper-parameters
    - Train it and check its performance metrics and decision boundary on train and test data
    - Diagnose possible overfitting and experiment with other settings
    """
    )

    st.markdown('''**Credit:** App built in `Python` + `Streamlit` by :red[**Akash Sharma** ] ''')
    st.write('---')


def dataset_selector():
    st.sidebar.subheader("Configure the Data")
    dataset_container = st.sidebar.expander("Create your own dataset", True)
    with dataset_container:
        dataset = st.selectbox("Choose a dataset", ("Moons", "Circles", "Blobs"))
        n_samples = st.number_input(
            "Number of Samples",
            min_value=50,
            max_value=1000,
            step=10,
            value=300,
        )

        train_noise = st.slider(
            "Set the noise (Train Data)",
            min_value=0.01,
            max_value=0.2,
            step=0.005,
            value=0.06,
        )
        test_noise = st.slider(
            "Set the noise (Test Data)",
            min_value=0.01,
            max_value=1.0,
            step=0.005,
            value=train_noise,
        )

        if dataset == "Blobs":
            n_classes = st.number_input("Centers", 2, 5, 2, 1)
        else:
            n_classes = None

    return dataset, n_samples, train_noise, test_noise, n_classes


def model_selector():
    st.sidebar.subheader("Train a Model")
    model_training_container = st.sidebar.expander("Choose Model and its Hyperparameters", True)
    with model_training_container:
        model_type = st.selectbox(
            "Choose a model",
            (
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "Neural Network",
                "K Nearest Neighbors",
                "Gaussian Naive Bayes",
                "SVC",
            ),
        )

        if model_type == "Logistic Regression":
            model = lr_param_selector()

        elif model_type == "Decision Tree":
            model = dt_param_selector()

        elif model_type == "Random Forest":
            model = rf_param_selector()

        elif model_type == "Neural Network":
            model = nn_param_selector()

        elif model_type == "K Nearest Neighbors":
            model = knn_param_selector()

        elif model_type == "Gaussian Naive Bayes":
            model = nb_param_selector()

        elif model_type == "SVC":
            model = svc_param_selector()

        elif model_type == "Gradient Boosting":
            model = gb_param_selector()

    return model_type, model


def generate_snippet(
    model, model_type, n_samples, train_noise, test_noise, dataset, degree
):
    train_noise = np.round(train_noise, 3)
    test_noise = np.round(test_noise, 3)

    model_text_rep = repr(model)
    model_import = model_imports[model_type]

    if degree > 1:
        feature_engineering = f"""
    >>> for d in range(2, {degree+1}):
    >>>     x_train = np.concatenate((x_train, x_train[:, 0] ** d, x_train[:, 1] ** d))
    >>>     x_test= np.concatenate((x_test, x_test[:, 0] ** d, x_test[:, 1] ** d))
    """

    if dataset == "Moons":
        dataset_import = "from sklearn.datasets import make_moons"
        train_data_def = (
            f"x_train, y_train = make_moons(n_samples={n_samples}, noise={train_noise})"
        )
        test_data_def = f"x_test, y_test = make_moons(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "Circles":
        dataset_import = "from sklearn.datasets import make_circles"
        train_data_def = f"x_train, y_train = make_circles(n_samples={n_samples}, noise={train_noise})"
        test_data_def = f"x_test, y_test = make_circles(n_samples={n_samples // 2}, noise={test_noise})"

    elif dataset == "Blobs":
        dataset_import = "from sklearn.datasets import make_blobs"
        train_data_def = f"x_train, y_train = make_blobs(n_samples={n_samples}, clusters=2, noise={train_noise* 47 + 0.57})"
        test_data_def = f"x_test, y_test = make_blobs(n_samples={n_samples // 2}, clusters=2, noise={test_noise* 47 + 0.57})"

    snippet = f"""
    >>> {dataset_import}
    >>> {model_import}
    >>> from sklearn.metrics import accuracy_score, f1_score

    >>> {train_data_def}
    >>> {test_data_def}
    {feature_engineering if degree > 1 else ''}    
    >>> model = {model_text_rep}
    >>> model.fit(x_train, y_train)
    
    >>> y_train_pred = model.predict(x_train)
    >>> y_test_pred = model.predict(x_test)
    >>> train_accuracy = accuracy_score(y_train, y_train_pred)
    >>> test_accuracy = accuracy_score(y_test, y_test_pred)
    """
    return snippet


def polynomial_degree_selector():
    return st.sidebar.number_input("Highest polynomial degree", 1, 10, 1, 1)
