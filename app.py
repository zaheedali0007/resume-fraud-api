from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the trained model and vectorizer
with open("resume_fraud_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    resume = data.get("resume")
    if not resume:
        return jsonify({"error": "No resume provided"}), 400

    vectorized = vectorizer.transform([resume])
    prediction = int(model.predict(vectorized)[0])
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
