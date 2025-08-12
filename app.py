from flask import Flask, request, render_template
import pickle

# Load model and vectorizer
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
except FileNotFoundError as e:
    print("❌ Error: Model or vectorizer file not found.")
    raise e
except Exception as e:
    print("❌ Error loading model/vectorizer.")
    raise e

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)
