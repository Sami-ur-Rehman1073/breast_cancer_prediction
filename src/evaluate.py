"""
evaluate.py

This module contains all evaluation-related functions for the
Breast Cancer Prediction project.

"""

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# Load Pipeline

def load_pipeline(filepath):
    """
    Load a trained pipeline.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    Pipeline
    """

    return joblib.load(filepath)


# Prediction

def predict_pipeline(
    pipeline,
    X_test,
):
    """
    Generate predictions using the trained pipeline.

    Returns
    -------
    predictions
    probabilities
    """

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    return predictions, probabilities



# Cross Validation

def perform_cross_validation(
    pipeline,
    X_train,
    y_train,
    cv=5,
):
    """
    Perform Stratified K-Fold Cross Validation.

    Returns
    -------
    pandas.DataFrame
    """

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    stratified_cv = StratifiedKFold(
        n_splits=cv,
        shuffle=True,
        random_state=42,
    )

    cv_results = cross_validate(
        estimator=pipeline,
        X=X_train,
        y=y_train,
        cv=stratified_cv,
        scoring=scoring,
        return_train_score=False,
    )

    return pd.DataFrame(cv_results)




# Average Cross Validation Results


def get_average_cv_scores(cv_results):
    """
    Calculate average cross-validation metrics.
    """

    metric_columns = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc",
    ]

    return (
        cv_results[metric_columns]
        .mean()
        .rename("Average")
    )