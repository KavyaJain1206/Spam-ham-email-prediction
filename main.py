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

# ------------------ NLTK downloads ------------------
nltk.download("punkt")
nltk.download("stopwords")

# ------------------ Load saved TF-IDF and model ------------------
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("bernoulli_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------ Preprocessing function ------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    tokens = word_tokenize(text)
    clean_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in clean_tokens]
    return " ".join(stemmed_tokens)

# ------------------ Explain function ------------------
def explain_prediction(text: str) -> dict:
    processed = preprocess_text(text)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)[0]

    spam_probs = model.feature_log_prob_[1]  # class 1 (spam)
    ham_probs = model.feature_log_prob_[0]   # class 0 (ham)
    contributions = dict(zip(tfidf.get_feature_names_out(), spam_probs - ham_probs))

    words = processed.split()
    word_scores = {word: contributions.get(word, 0) for word in words}
    top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "prediction": int(prediction),
        "top_contributing_words": top_words,
    }

# ------------------ FastAPI setup ------------------
app = FastAPI(title="Spam-Ham Detection API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React frontend if build exists
frontend_path = "frontend/build"
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"Warning: {frontend_path} not found. Frontend will not be served.")

# ------------------ Ping endpoint ------------------
@app.get("/ping")
def ping():
    return {"status": "alive"}

# ------------------ Pydantic model ------------------
class Message(BaseModel):
    text: str

# ------------------ API Endpoints ------------------
@app.post("/predict")
def predict_spam(message: Message):
    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message text is empty.")
    try:
        processed = preprocess_text(message.text)
        vector = tfidf.transform([processed])
        prediction = model.predict(vector)[0]
        return {"prediction": int(prediction), "label": "spam" if prediction == 1 else "ham"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain")
def explain_spam(message: Message):
    if not message.text.strip():
        raise HTTPException(status_code=400, detail="Message text is empty.")
    try:
        result = explain_prediction(message.text)
        result["label"] = "spam" if result["prediction"] == 1 else "ham"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ Run server ------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
