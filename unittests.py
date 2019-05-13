import pandas as pd
import pytest

from .model import load_data, preprocess_training_data, train_log_reg

##### model.py tests #####


def test_load_data_as_frame():
    valid_file = "data/bank-full.csv"
    assert type(load_data(valid_file)) == pd.DataFrame


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("bank.csv")


def test_load_data_invalid_delimeter():
    with pytest.raises(TypeError):
        load_data("data/bank-full.csv", 1.0203)
        load_data("data/bank-full.csv", 123)
        load_data("data/bank-full.csv", [])
        load_data("data/bank-full.csv", {})
        load_data("data/bank-full.csv", (1, 2, 3))


def test_preprocess_training_data_invalid_target():
    valid_data = load_data("data/bank-full.csv")
    invalid_data = pd.DataFrame(data={"a": 1234, "b": "hi", "c": 0.09}, index=[0])
    with pytest.raises(KeyError):
        preprocess_training_data(invalid_data, target="hello")
