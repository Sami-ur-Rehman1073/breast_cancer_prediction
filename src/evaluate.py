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


# Cross Validation Standard Deviation

def get_cv_std(cv_results):
    """
    Calculate standard deviation of cross-validation metrics.
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
        .std()
        .rename("Std Dev")
    )


# Test Metrics

def calculate_metrics(
    y_test,
    predictions,
    probabilities,
):
    """
    Calculate evaluation metrics.
    """

    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),

        "Precision": precision_score(
            y_test,
            predictions,
        ),

        "Recall": recall_score(
            y_test,
            predictions,
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
        ),

        "ROC AUC": roc_auc_score(
            y_test,
            probabilities,
        ),

    }

    return metrics


# Print Test Metrics

def print_test_metrics(metrics):
    """
    Print evaluation metrics.
    """

    for metric, value in metrics.items():

        print(f"{metric:<10}: {value:.4f}")


# Confusion Matrix

def plot_confusion_matrix(
    y_test,
    predictions,
):
    """
    Plot the confusion matrix.

    Parameters
    ----------
    y_test : pd.Series
    predictions : np.ndarray
    """

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"],
    )

    disp.plot(
        cmap="Blues",
        values_format="d",
    )

    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.show()




# Classification Report

def print_classification_report(
    y_test,
    predictions,
):
    """
    Print classification report.
    """

    print(classification_report(
        y_test,
        predictions,
    ))



# ==========================================================
# ROC Curve
# ==========================================================

def plot_roc_curve(
    pipeline,
    X_test,
    y_test,
):
    """
    Plot ROC Curve.
    """

    RocCurveDisplay.from_estimator(
        pipeline,
        X_test,
        y_test,
    )

    plt.title("ROC Curve")
    plt.show()



# Train vs Test Comparison

def compare_train_vs_test(
    pipeline,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Compare train and test performance.
    """

    train_predictions = pipeline.predict(X_train)
    test_predictions = pipeline.predict(X_test)

    train_probabilities = pipeline.predict_proba(
        X_train
    )[:, 1]

    test_probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    comparison = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC",
        ],

        "Training": [

            accuracy_score(
                y_train,
                train_predictions,
            ),

            precision_score(
                y_train,
                train_predictions,
            ),

            recall_score(
                y_train,
                train_predictions,
            ),

            f1_score(
                y_train,
                train_predictions,
            ),

            roc_auc_score(
                y_train,
                train_probabilities,
            ),
        ],

        "Test": [

            accuracy_score(
                y_test,
                test_predictions,
            ),

            precision_score(
                y_test,
                test_predictions,
            ),

            recall_score(
                y_test,
                test_predictions,
            ),

            f1_score(
                y_test,
                test_predictions,
            ),

            roc_auc_score(
                y_test,
                test_probabilities,
            ),
        ],
    })

    comparison["Difference"] = (
        comparison["Training"]
        - comparison["Test"]
    ).abs()

    return comparison


# Complete Evaluation

def evaluate_model(
    pipeline,
    X_test,
    y_test,
):
    """
    Perform complete model evaluation.
    """

    predictions, probabilities = predict_pipeline(
        pipeline,
        X_test,
    )

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    print("=" * 60)
    print("Test Performance")
    print("=" * 60)

    print_test_metrics(metrics)

    print()

    print("=" * 60)
    print("Classification Report")
    print("=" * 60)

    print_classification_report(
        y_test,
        predictions,
    )

    plot_confusion_matrix(
        y_test,
        predictions,
    )

    plot_roc_curve(
        pipeline,
        X_test,
        y_test,
    )

    return metrics



# Evaluate Multiple Models

def evaluate_multiple_models(
    trained_models,
    X_test,
    y_test,
):
    """
    Evaluate all trained models.
    """

    results = []

    for model_name, pipeline in trained_models.items():

        predictions, probabilities = predict_pipeline(
            pipeline,
            X_test,
        )

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities,
        )

        metrics["Model"] = model_name

        results.append(metrics)

    results = pd.DataFrame(results)

    results = results[[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC",
    ]]

    results = results.sort_values(
        by="ROC AUC",
        ascending=False,
    )

    return results


# Feature Importance

def plot_feature_importance(
    pipeline,
    feature_names,
    top_n=None,
):
    """
    Plot feature importance for the trained model.

    Supports:
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - XGBoost

    Parameters
    ----------
    pipeline : Pipeline
        Trained pipeline.

    feature_names : list
        List of feature names.

    top_n : int, optional
        Display only the top N features.
    """

    classifier = pipeline.named_steps["classifier"]

    # Logistic Regression

    if hasattr(classifier, "coef_"):

        importance = np.abs(
            classifier.coef_[0]
        )

    # Tree-Based Models

    elif hasattr(classifier, "feature_importances_"):

        importance = classifier.feature_importances_

    else:

        raise ValueError(
            f"{classifier.__class__.__name__} "
            "does not support feature importance."
        )

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False,
    )

    if top_n is not None:
        importance_df = importance_df.head(top_n)

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
    )

    plt.title("Feature Importance")

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.tight_layout()

    plt.show()

    return importance_df