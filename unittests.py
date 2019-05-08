import pytest
import pandas as pd
from .model import load_data


def test_load_data_as_frame():
    valid_file = "data/bank-full.csv"
    assert type(load_data(valid_file)) == pd.DataFrame


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("bank.csv")

