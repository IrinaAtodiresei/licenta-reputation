# sentiment_models.py
from typing import Dict
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline


class LogisticRegressionSentiment:
    """
    Model Logistic Regression 3-class (pozitiv / neutru / negativ).
    Pentru demo, antrenăm pe un mic dataset mock.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._is_fitted = False
        self._train_mock()

    def _train_mock(self):
        texts = [
            "I love this product, it is amazing and works great",
            "This is the worst thing I ever bought, I hate it",
            "It is okay, nothing special, just fine",
            "Absolutely fantastic experience, highly recommend",
            "Terrible quality and very disappointing",
            "Not bad, but could be better",
        ]
        labels = ["positive", "negative", "neutral", "positive", "negative", "neutral"]
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self._is_fitted = True

    def predict(self, text: str) -> Dict:
        if not self._is_fitted:
            self._train_mock()
        X = self.vectorizer.transform([text])
        proba = self.model.predict_proba(X)[0]
        label = self.model.classes_[np.argmax(proba)]
        return {
            "label": label,
            "proba": {cls: float(p) for cls, p in zip(self.model.classes_, proba)},
        }

    def word_contributions(self, text: str) -> pd.DataFrame:
        """Contribuția fiecărui cuvânt (coeficient * tf-idf)."""
        if not self._is_fitted:
            self._train_mock()

        tokens = text.split()
        X = self.vectorizer.transform([text])
        coefs = self.model.coef_[0]
        feature_names = self.vectorizer.get_feature_names_out()

        contribs = []
        for token in tokens:
            token_lower = token.lower().strip(".,!?")
            if token_lower in feature_names:
                idx = np.where(feature_names == token_lower)[0][0]
                value = X[0, idx]
                contrib = value * coefs[idx]
            else:
                contrib = 0.0
            contribs.append({"token": token, "contribution": float(contrib)})

        return pd.DataFrame(contribs)


logreg_model = LogisticRegressionSentiment()
vader_analyzer = SentimentIntensityAnalyzer()
transformer_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)


def predict_logreg(text: str) -> Dict:
    return logreg_model.predict(text)


def predict_logreg_word_contrib(text: str) -> pd.DataFrame:
    return logreg_model.word_contributions(text)


def predict_vader(text: str) -> Dict:
    scores = vader_analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"label": label, "compound": compound, "scores": scores}


def predict_transformer(text: str) -> Dict:
    out = transformer_pipeline(text)[0]
    label = out["label"].lower()
    score = float(out["score"])
    return {"label": label, "score": score}
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def predict_vader(text):
    """
    Returns:
    - label (positive/neutral/negative)
    - compound score
    - tokens (list)
    - scores (list)
    """

    vs = analyzer.polarity_scores(text)

    # Determine label
    compound = vs["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    # Token-level contributions
    tokens = text.split()
    scores = [analyzer.polarity_scores(tok)["compound"] for tok in tokens]

    return {
        "label": label,
        "compound": compound,
        "tokens": tokens,
        "scores": scores,
    }
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# ============================================================
# LOAD MODEL WITH ATTENTION ENABLED
# ============================================================
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment",
    output_attentions=True
)


def predict_transformer(text):
    """
    Returns:
    - label (positive/neutral/negative)
    - score (confidence)
    - tokens (list)
    - attention (matrix NxN)
    """

    # Tokenizare + tensori
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

    # Forward pass cu atenție
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    attentions = outputs.attentions  # listă: [layer][batch][head][seq][seq]

    # ============================================================
    # 1. Predicție sentiment
    # ============================================================
    probs = torch.softmax(logits, dim=-1)[0]
    score = float(torch.max(probs))
    label_id = int(torch.argmax(probs))

    # Modelul are 5 clase (1–5 stele)
    if label_id <= 1:
        label = "negative"
    elif label_id == 2:
        label = "neutral"
    else:
        label = "positive"

    # ============================================================
    # 2. Tokeni reali
    # ============================================================
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # ============================================================
    # 3. Atenție reală (ultimul layer, medie pe head‑uri)
    # ============================================================
    last_layer = attentions[-1]              # shape: [batch, heads, seq, seq]
    mean_attention = last_layer.mean(dim=1)  # medie pe head‑uri → [batch, seq, seq]
    attention_matrix = mean_attention[0].cpu().numpy()

    return {
        "label": label,
        "score": score,
        "tokens": tokens,
        "attention": attention_matrix,
    }
