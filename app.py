from flask import Flask


app = Flask()


@app.route("/predict/csv", methods=["POST"])
def csv_customer_likelihood(csv):
    pass


@app.route("predict/customer", methods=["POST"])
def json_customer_likelihood(customer_profile):
    pass


if __name__ == "__main__":
    app.run("localhost")
