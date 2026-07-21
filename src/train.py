"""
train.py

This module contains all training-related functions for the
Breast Cancer Prediction project.

"""


import time
import joblib
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Classification Models


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier




# Model Dictionary

def get_models(random_state: int = 42):
    """
    Returns all machine learning models used in the project.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            random_state=random_state,
            max_iter=1000
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state
        ),

        "Random Forest": RandomForestClassifier(
            random_state=random_state
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        ),

        "XGBoost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        ),

        "KNN": KNeighborsClassifier(),

        "SVM": SVC(
            probability=True,
            random_state=random_state
        ),
    }

    return models


# Pipeline Creation

def create_pipeline(preprocessor, model,random_state: int = 42):
    """
    Creates an imbalanced-learn pipeline consisting of:

    StandardScaler (inside ColumnTransformer)
            ↓
          SMOTE
            ↓
        ML Classifier
    """

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),

            (
                "smote",
                SMOTE(random_state=random_state),
            ),

            (
                "classifier",
                model,
            ),
        ]
    )

    return pipeline


# Fit Pipeline


def fit_pipeline(
    pipeline,
    X_train,
    y_train,
):
    """
    Fits the pipeline on the training data.

    Returns
    -------
    pipeline
        Trained pipeline.
    """

    start_time = time.time()

    pipeline.fit(
        X_train,
        y_train,
    )

    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds.")

    return pipeline


# Prediction

def predict_pipeline(
    pipeline,
    X_test,
):
    """
    Generate predictions and prediction probabilities.
    """

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    return predictions, probabilities