import json

import joblib
import numpy as np
import pandas as pd

from model import load_data, preprocess_training_data
from predict_utils import TRAINING_COLUMNS


def create_user_profile(user: dict) -> pd.DataFrame:
    """Returns a pd.DataFrame representation of a user from a JSON object.

    Arguments:
        user {dict} -- JSON object posted via the /predict/customer route.

    Returns:
        user {pd.DataFrame} -- User profile for further analysis.
    """
    user = pd.DataFrame.from_dict([user])
    return user


def predict_user(user_profile: pd.DataFrame, round_prec: int = 3) -> str:
    """Predicts a user's likelihood of subscribing to a product based on a
    pd.DataFrame representation of their user profile. This profile is then
    re-indexed with the columns used during training (TRAINING_COLUMNS) to
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


def predict_frame(frame: pd.DataFrame) -> str:
    """Takes in a pd.DataFrame from an uploaded .csv file.

    [For the purposes of the exercise] -- Drops the Y column for predictions.

    Reindexes the DataFrame according to a one-hot encoded representation of
    all of the columns.

    Applies the loaded logistic regression model across the reshaped numpy
    array representation of each row, creating a new pd.Series column in the
    process. Rounds the values for nicer viewing.

    ---
    The reason why reindex_frame is used for the application and is not
    returned is to reduce the load on to_json() when it returns the resulting
    JSON string from the predictions (instead of returning all of the OHE cols
    as well).
    ---

    Finally, returns the pd.DataFrame as a JSON string back to the /predict/csv
    API route.

    Arguments:
        frame {pd.DataFrame} -- pd.DataFrame of the uploaded .csv file.

    Returns:
        str -- Record-oriented JSON string containing all original columns

    """
    model = joblib.load("models/LogReg.pkl")
    frame = load_data(frame)
    frame.drop("y", axis=1, inplace=True)
    reindex_frame = frame.reindex(
        columns=TRAINING_COLUMNS.columns, fill_value=0
    )
    frame["yes_prob"] = reindex_frame.apply(
        lambda x: model.predict_proba(np.array(x).reshape(1, -1))[0][1], axis=1
    ).round(3)
    return frame.to_json(orient="records")
