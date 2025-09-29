import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import os
import sys

# ------------------ NLTK downloads ------------------

try:
    print("Downloading NLTK resources...")
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    print("NLTK downloads complete.")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    sys.exit(1)


# ------------------ Load saved TF-IDF and model ------------------

# IMPORTANT: Ensure 'tfidf_vectorizer.pkl' and 'bernoulli_model.pkl'
# (containing the trained scikit-learn objects) are in the same directory.
try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    print("TF-IDF Vectorizer loaded successfully.")

    with open("bernoulli_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("Bernoulli Naive Bayes Model loaded successfully.")

except FileNotFoundError:
    print("\n[FATAL ERROR] Model or Vectorizer file not found.")
    print("Please ensure 'tfidf_vectorizer.pkl' and 'bernoulli_model.pkl' are present.")
    sys.exit(1)
except Exception as e:
    print(f"\n[FATAL ERROR] Failed to load model or vectorizer: {e}")
    sys.exit(1)


# ------------------ Preprocessing function ------------------

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """Cleans, tokenizes, removes stop words, and stems the input text."""
    text = text.lower()
    tokens = word_tokenize(text)
    # Filter for alphanumeric tokens and remove stop words
    clean_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    # Apply stemming
    stemmed_tokens = [stemmer.stem(token) for token in clean_tokens]
    return " ".join(stemmed_tokens)

# ------------------ Explain function ------------------

def explain_prediction(text: str) -> dict:
    """
    Predicts the label and provides the top 5 words contributing to the decision.
    Uses log-probability difference (log P(word|spam) - log P(word|ham)) as contribution score.
    """
    processed = preprocess_text(text)
    vector = tfidf.transform([processed])
    
    # Predict the label
    prediction = model.predict(vector)[0]

    # Calculate contribution scores using log probabilities from the Naive Bayes model
    # Naive Bayes stores log probabilities for each feature (word) per class.
    # spam_probs[i] = log P(feature_i | class=spam)
    # ham_probs[i] = log P(feature_i | class=ham)
    
    # Note: BernoulliNB stores these probabilities in feature_log_prob_
    if hasattr(model, 'feature_log_prob_') and model.feature_log_prob_.shape[0] == 2:
        spam_probs = model.feature_log_prob_[1]  # class 1 (spam)
        ham_probs = model.feature_log_prob_[0]   # class 0 (ham)
        
        # Calculate the difference: Positive scores indicate higher association with spam (class 1)
        contributions = dict(zip(tfidf.get_feature_names_out(), spam_probs - ham_probs))
    else:
        # Fallback if model structure is unexpected
        contributions = {}
        print("Warning: Could not access feature_log_prob_ for explanation.")


    words = processed.split()
    # Map the stemmed words back to their contribution scores
    word_scores = {word: contributions.get(word, 0) for word in words}
    
    # Sort by the absolute value of the score to find the most influential words (in either direction)
    top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "prediction": int(prediction),
        "top_contributing_words": [{"word": w, "score": s} for w, s in top_words],
    }

# ------------------ FastAPI setup ------------------

app = FastAPI(title="Spam-Ham Detection API")

# Allow CORS for development/cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Ping endpoint ------------------

@app.get("/ping")
def ping():
    """Simple health check endpoint."""
    return {"status": "alive", "model_loaded": True}

# ------------------ Pydantic model ------------------

class Message(BaseModel):
    """Data model for incoming text message."""
    text: str

# ------------------ API Endpoints ------------------

@app.post("/predict")
def predict_spam(message: Message):
    """Endpoint for simple spam/ham prediction."""
    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message text is empty.")
    
    processed = preprocess_text(message.text)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)[0]
    
    return {
        "prediction": int(prediction), 
        "label": "spam" if prediction == 1 else "ham"
    }

@app.post("/explain")
def explain_spam(message: Message):
    """Endpoint for prediction with word-level explanation."""
    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message text is empty.")
        
    result = explain_prediction(message.text)
    result["label"] = "spam" if result["prediction"] == 1 else "ham"
    return result

# ------------------ Serve React frontend LAST ------------------

# This section serves the static files (HTML, CSS, JS) of a frontend application.
# It MUST be placed last so it doesn't override the API endpoints above.
frontend_path = "frontend/build"
if os.path.isdir(frontend_path):
    print(f"Serving static files from /{frontend_path}")
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Warning: Directory '{frontend_path}' not found. Frontend will not be served.")

# ------------------ Run server ------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    # Note: In a production environment, use a separate WSGI server (like gunicorn) 
    # to manage uvicorn worker processes.
    print(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
