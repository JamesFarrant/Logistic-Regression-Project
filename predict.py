# TODO: Create an API endpoint which accepts relevant parameters from a
# potential customer and returns the likelihood of them
# also subscribing to the product
from model import preprocess_training_data
import pandas as pd
import joblib


def create_user_profile(user):
    pass


def predict_user(
    user_profile: pd.DataFrame, model="models/LogReg.pkl"
) -> float:
    """[summary]

    Arguments:
        user_profile {[type]} -- [description]
    """
    model = joblib.load(model)
    prediction = model.predict_proba(user_profile)
