import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


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
    pass


def train_model(training_data: pd.DataFrame) -> LogisticRegression:
    """[summary]

    Arguments:
        training_data {pd.DataFrame} -- [description]

    Returns:
        sklearn.linear_model.LogisticRegression -- [description]
    """
    log_reg = LogisticRegression()
    x_train, x_test, y_train, y_test = train_test_split(
        training_data[
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
        ],
        training_data["y"]
    )
    print(x_train, x_test, y_train, y_test)

print(train_model(load_data("data/bank-full.csv")))
