import torch.nn as nn
from flask import render_template, request, jsonify
from app import app
from app import model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news_headline = request.form['news_headline']

    predicti = model.prediction(news_headline)

    return jsonify({'prediction': predicti})