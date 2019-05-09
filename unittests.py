import pandas as pd
import pytest

from .model import load_data, preprocess_training_data, train_log_reg

##### model.py tests #####
# ----- load_data tests -----


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


# ----- preprocess_training_data tests -----

# ----- train_log_reg tests -----


##### predict.py tests #####

##### app.py tests #####