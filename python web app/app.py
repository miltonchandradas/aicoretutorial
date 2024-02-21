from flask import Flask

# Creates Flask serving engine
app = Flask(__name__)


@app.route("/v2/greeting", methods=["GET"])
def greeting():
    return "Flask Code: Hello world..."


