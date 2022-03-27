from flask import Flask, request, jsonify
from fastai.basics import load_learner
from fastai.vision.core import *
from flask_cors import CORS,cross_origin
from pathlib import Path

app = Flask(__name__)
CORS(app, support_credentials=True)

path = Path()

# load the learner
learn = load_learner(path/'100items_model_.pkl')
classes = learn.dls.vocab

# function to take image and return prediction
def predict_single(img_file):
    
    prediction = learn.predict(PILImage.create(img_file))
    probs_list = prediction[2].numpy()
    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }


# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

if __name__ == '__main__':
    app.run()