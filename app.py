import pickle
import flask
import json

from flask import Flask, request, jsonify


## loading the model
#model_pickle = open("./question_answer.pkl", 'rb')
#model = pickle.load(model_pickle)
model = pipeline("question-answering")

app = Flask(__name__)


# helper function to read a text file
def read_text_file(fname):
    with open(fname) as f:
        text = "".join(f.readlines()).replace("\n", " ")
    return text


# defining the function which will make the prediction using the data which the user inputs 
@app.route('/predict', methods = ['POST'])
def prediction():
    """
    Sample Input Json - 
    {
        "query": "What is the minimum period of hospitalization?",
        "policy_fname": "max_bupa_health_recharge.txt"
    }

    """

    # Pre-processing user input
    query = json.loads(request.data)["query"]
    context = read_text_file(json.loads(request.data)["policy_fname"])

    # Making predictions
    result = jsonify(model(question=query, context=text))
    return result


@app.route('/ping', methods=['GET'])
def ping():
    return "Pinging Model!!"
