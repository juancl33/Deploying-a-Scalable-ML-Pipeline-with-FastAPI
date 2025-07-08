import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
# TODO: add necessary import

# Load and process some sample data for testing
@pytest.fixture
# TODO: implement the first test. Change the function name and input as needed
def sample_data():
    df = pd.read_csv("data/census.csv")
    cat_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]
    X, y, encoder, lb = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    return X, y


# TODO: implement the second test. Change the function name and input as needed
def test_train_model_returns_model(sample_data):
    """
    Test that train_model returns a fitted RandomForestClassifier.
    """
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


# TODO: implement the third test. Change the function name and input as needed
def test_inference_shape_matches(sample_data):
    """
    Test that inference returns predictions with the same length as input rows.
    """
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == X.shape[0]

def test_compute_model_metrics_output():
    """
    Test compute_model_metrics returns precision, recall, and fbeta as floats
    """
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1])
    p, r, f = compute_model_metrics(y_true, y_pred)
    assert all(isinstance(metric, float) for metric in (p, r, f))

