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


