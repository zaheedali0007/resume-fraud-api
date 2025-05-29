from flask_cors import CORS
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load the vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("resume_fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return 'Resume Fraud Detection API is live! Use POST /predict with JSON {"resume": "..."}'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    resume_text = data.get('resume', '')

    if not resume_text.strip():
        return jsonify({'error': 'Resume content is empty'}), 400

    vectorized = vectorizer.transform([resume_text])
    prediction = model.predict(vectorized)[0]
    label = 'Genuine' if prediction == 0 else 'Fraudulent'
    return jsonify({'prediction': label})

if __name__ == '__main__':
    app.run(debug=True)
