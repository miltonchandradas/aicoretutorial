import pickle
import numpy as np
from flask import Flask, request

# Creates Flask serving engine
app = Flask(__name__)

model = None
appHasRunBefore = False
flower = ""
sepal_length = ""
sepal_width = ""
petal_length = ""
petal_width = ""


@app.before_request
def init():
    """
    Load model else crash, deployment will not start
    """
    global model
    global appHasRunBefore

    if not appHasRunBefore:
        # All the model files will be read from /mnt/models
        model = pickle.load(open('/mnt/models/model.pkl', 'rb'))
        # model = pickle.load(open('model.pkl', 'rb'))
        appHasRunBefore = True
        return None


@app.route("/v2/greet", methods=["GET"])
def status():
    global model
    if model is None:
        return "Flask Code: Model was not loaded."
    else:
        return "Flask Code: Model loaded successfully."


@app.route("/v2/predict", methods=["POST"])
def predict():
    global model
    global flower
    global petal_length
    global petal_width
    global sepal_length
    global sepal_width
    
    print("Docker image version is 4.0")

    if model is None:
        return "Flask Code: Model was not loaded."
    else:
        query = dict(request.json)
        sepal_length = query["SepalLengthCm"]
        sepal_width = query["SepalWidthCm"]
        petal_length = query["PetalLengthCm"]
        petal_width = query["PetalWidthCm"]
        attributes = [sepal_length, sepal_width, petal_length, petal_width]
        print("Attributes: ", [attributes])
        prediction = model.predict(
            # (trailing comma) <,> to make batch with 1 observation
            [attributes]
        )
        
        if str(prediction) == "['Setosa']":
            flower = "Setosa"
        elif str(prediction) == "['Versicolor']":
            flower = "Veriscolor"
        elif str(prediction) == "['Virginica']":
            flower = "Virginica"
        else:
            flower = "Unknown"

        return {"attributes": {"SepalLengthCm": sepal_length, "SepalWidthCm": sepal_width, "PetalLengthCm": petal_length, "PetalWidthCm": petal_width}, "flower": flower}


if __name__ == "__main__":
    print("Serving Initializing")
    init()
    print("Serving Started")
    app.run(host="0.0.0.0", debug=True, port=9001)
    # app.run(debug=True)
