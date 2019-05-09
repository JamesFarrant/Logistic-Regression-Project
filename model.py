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


def preprocess_training_data(
    training_data: pd.DataFrame, target="y"
) -> pd.DataFrame:
    """Converts loaded .csv data into a one-hot-encoded format suitable for
    scikit-learn and deep learning models using pd.get_dummies.

    Arguments:
        training_data {pd.DataFrame} -- Raw training data
                                        represented as a pd.DataFrame

    Keyword Arguments:
        target {str} -- The target column for a model to predict
                        during training and when deployed.

    Returns:
        pd.DataFrame -- One-hot encoded representation of training_data.
    """
    training_data.loc[training_data[target] == "yes", target] = 1
    training_data.loc[training_data[target] == "no", target] = 0
    y = training_data[target]
    training_data.drop([target], axis=1, inplace=True)
    training_data = pd.get_dummies(training_data)
    training_data = pd.concat([training_data, y], axis=1)
    return training_data


def train_log_reg(training_data: pd.DataFrame, model_name="LogReg") -> str:
    """Trains an sklearn.linear_model.LogisticRegression model based on data
    represented by training_data in pd.DataFrame format.

    Arguments:
        training_data {pd.DataFrame} -- A preprocessed pd.DataFrame created
                                        from training data represented as a
                                        .csv file.

    Keyword Arguments:
        model_name {str} -- The prefix of the model to use when saving it as a
                            .pkl file.

    Returns:
        {str} -- Statement showing model was successfully trained and saved.
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
