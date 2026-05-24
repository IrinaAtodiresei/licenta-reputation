import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
    print("VADER OK")
except:
    print("Downloading VADER lexicon...")
    nltk.download("vader_lexicon")
