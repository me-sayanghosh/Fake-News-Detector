import os
from flask import Flask, request, render_template
import pickle

BASE_DIR = os.path.dirname(__file__)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
)

# Load model and vectorizer
try:
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as model_file:
        model = pickle.load(model_file)
    with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError as e:
    print("❌ Error: Model or vectorizer file not found.")
    raise e

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news = request.form['news']
        if not news.strip():
            return render_template('index.html', prediction="⚠️ Please enter some news content.")
        data = vectorizer.transform([news])
        prediction = model.predict(data)[0]
        label = "✅ Real News" if prediction == 1 else "❌ Fake News"
        return render_template('index.html', prediction=f'Result: {label}')
    except Exception as e:
        return render_template('index.html', prediction=f"❌ Error: {str(e)}")
