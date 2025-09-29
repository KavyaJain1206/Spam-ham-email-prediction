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
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    print(f"NLTK download error: {e}")
    sys.exit(1)

# ------------------ Load models ------------------
try:
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("bernoulli_model.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Failed to load model/vectorizer: {e}")
    sys.exit(1)

# ------------------ Preprocessing ------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    clean_tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    stemmed_tokens = [stemmer.stem(t) for t in clean_tokens]
    return " ".join(stemmed_tokens)

def explain_prediction(text: str) -> dict:
    processed = preprocess_text(text)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)[0]

    if hasattr(model, 'feature_log_prob_'):
        spam_probs = model.feature_log_prob_[1]
        ham_probs = model.feature_log_prob_[0]
        contributions = dict(zip(tfidf.get_feature_names_out(), spam_probs - ham_probs))
    else:
        contributions = {}

    words = processed.split()
    word_scores = {word: contributions.get(word, 0) for word in words}
    top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    # Convert to objects to match React chart mapping
    return {
        "prediction": int(prediction),
        "top_contributing_words": [{"word": w, "score": s} for w, s in top_words]
    }

# ------------------ FastAPI ------------------
app = FastAPI(title="Spam-Ham Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all for Render frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Pydantic ------------------
class Message(BaseModel):
    text: str

# ------------------ API Endpoints ------------------
@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.post("/predict")
def predict_spam(message: Message):
    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message is empty")
    processed = preprocess_text(message.text)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)[0]
    return {"prediction": int(prediction), "label": "spam" if prediction == 1 else "ham"}

@app.post("/explain")
def explain_spam(message: Message):
    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message is empty")
    result = explain_prediction(message.text)
    result["label"] = "spam" if result["prediction"] == 1 else "ham"
    return result

# ------------------ Serve React frontend ------------------
frontend_path = "frontend/build"
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Warning: Frontend build directory '{frontend_path}' not found.")

# ------------------ Run ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
