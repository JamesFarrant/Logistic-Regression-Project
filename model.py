import pandas as pd
from sklearn.linear_model import LogisticRegression


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
