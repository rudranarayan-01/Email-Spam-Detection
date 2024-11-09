from flask import Flask, request, render_template, jsonify
import pickle
import os

app = Flask(__name__)

# Load the classifier model and vectorizer
with open("models/spam_classifier.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("models/vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the email text from the form
        email_text = request.form["email_text"]
        
        # Vectorize the input text
        email_vector = vectorizer.transform([email_text])
        
        # Predict the label (0 = ham, 1 = spam)
        prediction = model.predict(email_vector)[0]
        result = "Spam" if prediction == 1 else "Ham"
        
        # Render the result
        return render_template("index.html", email_text=email_text, prediction=result)
    
    return render_template("index.html")

# Optional: API endpoint for programmatic access
@app.route("/api/classify", methods=["POST"])
def classify_email():
    data = request.get_json()
    email_text = data.get("email_text", "")

    # Vectorize and classify the input text
    email_vector = vectorizer.transform([email_text])
    prediction = model.predict(email_vector)[0]
    result = "Spam" if prediction == 1 else "Ham"
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
