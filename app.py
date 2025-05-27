from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("resume_fraud_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'resume' not in data:
        return jsonify({'error': 'Missing "resume" field'}), 400

    resume_text = data['resume']
    features = vectorizer.transform([resume_text])
    prediction = model.predict(features)[0]

    label = "Genuine" if prediction == 0 else "Fraudulent"

    return jsonify({'prediction': label})

@app.route('/')
def home():
    return 'Resume Fraud Detection API is live! Use POST /predict with JSON {"resume": "..."}'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
