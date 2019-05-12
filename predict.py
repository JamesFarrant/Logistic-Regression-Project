from artificialio.model import load_data, preprocess_training_data
from .predict_utils import TRAINING_COLUMNS
import pandas as pd
import joblib
import json


test_user = {
    "age": 58,
    "job": "management",
    "education": "tertiary",
    "default": "no",
    "balance": 2143,
    "housing": "yes",
    "loan": "no",
    "contact": "unknown",
    "day": 5,
    "month": "may",
    "duration": 261,
    "campgaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}


def create_user_profile(user: dict) -> pd.DataFrame:
    """[summary]

    Arguments:
        user {[type]} -- [description]
    """
    user = pd.DataFrame.from_dict([user])
    return user


def predict_user(user_profile: pd.DataFrame, round_prec=3) -> float:
    """Predicts a user's likelihood of subscribing to a product based on a
    pd.DataFrame representation of their user profile. This profile is then
    reindexed with the columns used during training (TRAINING_COLUMNS) to
    allow for the LogisticRegression model to return predictions for it.

    Arguments:
        user_profile {pd.DataFrame} -- The user's profile represented as a
                                    pd.DataFrame.

    Keyword Arguments:
        round_prec {int} -- Rounding precision of the prediction (default: {3})

    Returns:
        {str} -- JSON string representation of model "yes_prob" prediction:
                likelihood user _will_ subscribe to product based on profile.
    """
    model = joblib.load("models/LogReg.pkl")
    user = user_profile.reindex(columns=TRAINING_COLUMNS.columns, fill_value=0)
    prediction = model.predict_proba(user)
    pred_dict = {
        "no_prob": round(prediction[0][0], round_prec),
        "yes_prob": round(prediction[0][1], round_prec),
    }
    return json.dumps(pred_dict["yes_prob"])
