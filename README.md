# artificialio


### Example requests (recommend Postman):

#### POST localhost:5000/predict/customer (type: JSON application/json)
```python
{
    "age": 33,
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
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
}
```
Example output: 0.587 - "yes" probability.

#### POST localhost:5000/predict/csv (form-data) key: file type: file
I would recommend using test.csv as the full dataset takes a while. Looks like to_json is the bottleneck here.


#### Next Steps

1. Prevent incorrect number of columns when predicting on model.
2. Find a better way of converting .csv prediction output to JSON - parallel processing? Dask? Swifter? Write to file first then output that instead?
3. More tests on custom-built functions.
4. Potentially look for a more elegant way to reindex the columns during prediction
5. Try different ML models and record train/test performance in a log file.


I really enjoyed the assignment and look forward to speaking with you soon!