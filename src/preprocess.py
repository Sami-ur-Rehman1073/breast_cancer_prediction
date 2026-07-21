"""
preprocess.py

This module contains all preprocessing-related functions for the
Breast Cancer Prediction project.

"""

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================================
# Train-Test Split
# ==========================================================

def split_data(
    X,
    y,
    test_size=0.2,
    random_state=42,
):
    """
    Split the dataset into training and testing sets.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.

    y : pandas.Series
        Target variable.

    test_size : float
        Proportion of test data.

    random_state : int
        Random seed.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# ==========================================================
# Column Transformer
# ==========================================================

def create_preprocessor(feature_names):
    """
    Create a preprocessing pipeline.

    Parameters
    ----------
    feature_names : list
        List of numerical feature names.

    Returns
    -------
    ColumnTransformer
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                feature_names,
            )
        ],
        remainder="drop",
    )

    return preprocessor


# ==========================================================
# Save Preprocessor
# ==========================================================

def save_preprocessor(preprocessor, filepath):
    """
    Save preprocessing object.

    Parameters
    ----------
    preprocessor : ColumnTransformer

    filepath : str
    """

    joblib.dump(preprocessor, filepath)


# ==========================================================
# Load Preprocessor
# ==========================================================

def load_preprocessor(filepath):
    """
    Load preprocessing object.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    ColumnTransformer
    """

    return joblib.load(filepath)


# ==========================================================
# Save Train/Test Data
# ==========================================================

def save_train_test_data(
    X_train,
    X_test,
    y_train,
    y_test,
    filepath,
):
    """
    Save train-test split using Joblib.
    """

    joblib.dump(
        {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        filepath,
    )


# ==========================================================
# Load Train/Test Data
# ==========================================================

def load_train_test_data(filepath):
    """
    Load saved train-test split.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    dict
    """

    return joblib.load(filepath)


# ==========================================================
# Complete Preprocessing Workflow
# ==========================================================

def preprocess_data(
    X,
    y,
    preprocessor_path,
    split_data_path,
    test_size=0.2,
    random_state=42,
):
    """
    Complete preprocessing workflow.

    Steps
    -----
    1. Train-Test Split
    2. Create ColumnTransformer
    3. Save Preprocessor
    4. Save Train-Test Data

    Returns
    -------
    tuple
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    """

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = create_preprocessor(
        feature_names=X.columns.tolist()
    )

    save_preprocessor(
        preprocessor,
        preprocessor_path,
    )

    save_train_test_data(
        X_train,
        X_test,
        y_train,
        y_test,
        split_data_path,
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )