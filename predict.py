# TODO: Create an API endpoint which accepts relevant parameters from a
# potential customer and returns the likelihood of them
# also subscribing to the product
from model import load_data, preprocess_training_data
from predict_utils import TRAINING_COLUMNS
import pandas as pd
import joblib
import json


# TODO: Data formatting when submitted as JSON
# This particular profile _should_ have a high probability of 0
# (not subscribing)

# TODO: Preprocess input JSON to be the same OHE format as the trained model
# ahead of prediction
# simply retrieve (and store, if needed) the list of columns after the training OHE output.
# Then run pd.get_dummies on the test set/example. Loop through the output test OHE columns,
# drop those that do not appear in the training OHE and add those that are missing in test OHE filled with zeros.
test_user = json.dumps(
    {
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
)


'''
def create_user_profile(user):
    """[summary]

    Arguments:
        user {[type]} -- [description]
    """
    user = pd.read_json(user, )
    return user


print(create_user_profile(test_user))
'''


def predict_user(
    user_profile: pd.DataFrame, model="models/LogReg.pkl"
) -> float:
    """[summary]

    Arguments:
        user_profile {[type]} -- [description]
    """
    model = joblib.load(model)
    user = pd.read_csv("data/bank-full.csv", delimiter=";").head(1)[
        [
            "age",
            "job",
            "marital",
            "education",
            "default",
            "balance",
            "housing",
            "loan",
            "contact",
            "day",
            "month",
            "duration",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
        ]
    ]
    user = user.reindex(columns=TRAINING_COLUMNS.columns, fill_value=0)
    print(user)
    prediction = model.predict_proba(user)
    print(prediction)


print(predict_user(""))
