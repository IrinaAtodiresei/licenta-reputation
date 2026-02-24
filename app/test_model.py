import joblib
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent   # urcă din app -> LICENTA
model = joblib.load(BASE / "models" / "sentiment_lr_sent140.joblib")
tfidf = joblib.load(BASE / "models" / "tfidf_sent140.joblib")

texts = [
    "I love this phone, it's amazing and fast.",
    "This is garbage, worst experience ever."
]

X = tfidf.transform(texts)
pred = model.predict(X)
proba = model.predict_proba(X)

print("pred:", pred)    # 1=pos, 0=neg
print("proba:", proba)  # [P(neg), P(pos)]