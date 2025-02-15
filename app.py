from flask import Flask, request, render_template
import joblib
import numpy as np

tfidf = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("recommendation_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    processed_review = tfidf.transform([review]).toarray()
    prediction = model.predict(processed_review)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
