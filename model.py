import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(csv_path: str, delimiter: str = ";") -> pd.DataFrame:
    """Loads data from a .csv file into a pd.DataFrame using pd.read_csv.

    Arguments:
        csv_path {str} -- File path to the .csv file.

    Keyword Arguments:
        delimiter {str} -- Separation value within the file. (default: {";"})

    Returns:
        pd.DataFrame -- pd.DataFrame for further processing.
    """
    return pd.read_csv(csv_path, delimiter=delimiter)


def preprocess_training_data(training_data: pd.DataFrame) -> pd.DataFrame:
    """[summary]

    Arguments:
        training_data {pd.DataFrame} -- [description]

    Returns:
        pd.DataFrame -- [description]
    """
    training_data.loc[training_data["y"] == "yes", "y"] = 1
    training_data.loc[training_data["y"] == "no", "y"] = 0
    y = training_data["y"]
    training_data.drop(["y"], axis=1, inplace=True)
    training_data = pd.get_dummies(training_data)
    training_data = pd.concat([training_data, y], axis=1)
    return training_data


def train_model(training_data: pd.DataFrame, model_name="LogReg") -> str:
    """[summary]

    Arguments:
        training_data {pd.DataFrame} -- [description]

    Returns:
        sklearn.linear_model.LogisticRegression -- [description]
    """
    model = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(
        training_data.loc[:, training_data.columns != "y"], training_data["y"]
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("--- Training results ---")
    print(classification_report(y_test, predictions))
    print(f"Saving model {model_name} with parameters: {model}...")
    joblib.dump(model, f"models/{model_name}.pkl", compress=3)
    return (
        f"Successfully trained and saved model {model_name} as "
        f"{model_name}.pkl!"
    )


# print(train_model(preprocess_training_data(load_data("data/bank-full.csv"))))
