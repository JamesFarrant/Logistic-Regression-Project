from flask import Flask, Response, request
from model import load_data
from predict import create_user_profile, predict_user, predict_frame

app = Flask(__name__)


@app.route("/predict/csv", methods=["POST"])
def csv_customer_likelihood():
    if request.method == "POST":
        file = request.files["file"]
        return Response(predict_frame(file))


@app.route("/predict/customer", methods=["POST"])
def json_customer_likelihood():
    if request.method == "POST":
        customer_profile = request.get_json()
        return Response(predict_user(create_user_profile(customer_profile)))


if __name__ == "__main__":
    app.run("localhost")
