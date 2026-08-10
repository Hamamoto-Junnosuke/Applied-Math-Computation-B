"""
Core definition used throughout this project.

    I_Y(S) = L(f_empty) - L(f_S)

where f_S is a linear regression trained on the feature set S,
f_empty is the constant predictor (training mean), and L is the
mean squared error measured on held-out test data.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

SEED = 0
TEST_SIZE = 0.3


def load_data():
    """Return the California Housing features (DataFrame) and price (Series)."""
    dataset = fetch_california_housing(as_frame=True)
    return dataset.data, dataset.target


def information(features, target, table):
    """Compute I_Y(S) and the baseline loss L(f_empty).

    Parameters
    ----------
    features : list of str   feature names forming the set S (may be empty)
    target   : pd.Series     the prediction target Y
    table    : pd.DataFrame  table the feature names refer to

    Returns
    -------
    (info, baseline) : the information value and L(f_empty)
    """
    y = np.asarray(target, dtype=float)
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED
    )
    y_train, y_test = y[train_idx], y[test_idx]

    # Predictor using no features at all: always output the training mean.
    baseline = mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))
    if not features:
        return 0.0, baseline

    x = table[features].to_numpy(dtype=float)
    model = LinearRegression().fit(x[train_idx], y_train)
    loss = mean_squared_error(y_test, model.predict(x[test_idx]))
    return baseline - loss, baseline


def normalized_information(features, target, table):
    """I_Y(S) divided by L(f_empty).

    I_Y(S) carries the squared unit of Y, so values obtained for different
    targets cannot be compared directly. Dividing by the baseline loss makes
    the quantity dimensionless (the fraction of the baseline error removed)
    and therefore comparable across targets.
    """
    info, baseline = information(features, target, table)
    return info / baseline
