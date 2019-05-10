from flask import Flask, Response, request
from predict import create_user_profile, predict_user

app = Flask(__name__)


@app.route("/predict/csv", methods=["POST"])
def csv_customer_likelihood(csv):
    pass


@app.route("/predict/customer", methods=["POST"])
def json_customer_likelihood():
    customer_profile = request.get_json()
    return customer_profile


if __name__ == "__main__":
    app.run("localhost")
