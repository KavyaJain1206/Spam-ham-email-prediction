import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# ------------------ Load saved TF-IDF and model ------------------
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("bernoulli_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------ Preprocessing function ------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    clean_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    stemmed_tokens = [stemmer.stem(token) for token in clean_tokens]
    return " ".join(stemmed_tokens)  # join back for vectorizer

# ------------------ Explain function ------------------
def explain_prediction(text):
    processed = preprocess_text(text)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)[0]
    
    # Get feature contributions
    spam_probs = model.feature_log_prob_[1]  # class 1 (spam)
    ham_probs = model.feature_log_prob_[0]   # class 0 (ham)
    contributions = dict(zip(tfidf.get_feature_names_out(), spam_probs - ham_probs))
    
    # Score each word in input
    words = processed.split()
    word_scores = {word: contributions.get(word, 0) for word in words}
    top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    return {
        "prediction": int(prediction),  # 1 = spam, 0 = ham
        "top_contributing_words": top_words
    }

# ------------------ API setup ------------------
app = FastAPI(title="Spam-Ham Detection API")

class Message(BaseModel):
    text: str

@app.post("/predict")
def predict_spam(message: Message):
    processed = preprocess_text(message.text)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)[0]
    return {"prediction": int(prediction), "label": "spam" if prediction == 1 else "ham"}

@app.post("/explain")
def explain_spam(message: Message):
    result = explain_prediction(message.text)
    result["label"] = "spam" if result["prediction"] == 1 else "ham"
    return result

# ------------------ Run ------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
