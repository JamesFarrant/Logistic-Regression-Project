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

