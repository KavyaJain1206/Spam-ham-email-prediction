import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

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

    # Feature contributions
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

# Allow React frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for stricter security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Pydantic model ------------------
class Message(BaseModel):
    text: str

# ------------------ Endpoints ------------------
@app.post("/predict")
def predict_spam(message: Message):
    if not message.text.strip():
        return {"error": "Message text is empty."}

    processed = preprocess_text(message.text)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)[0]
    return {"prediction": int(prediction), "label": "spam" if prediction == 1 else "ham"}

@app.post("/explain")
def explain_spam(message: Message):
    if not message.text.strip():
        return {"error": "Message text is empty."}

    result = explain_prediction(message.text)
    result["label"] = "spam" if result["prediction"] == 1 else "ham"
    return result

# ------------------ Run server ------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
