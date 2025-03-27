import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def transform_text(text):
    # Download necessary NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')
    # Initialize the stemmer
    ps = PorterStemmer()
    stop_words = set(stopwords.words('english'))  # Convert to set for faster lookups
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Removing special characters, stop words, and punctuation + Stemming
    processed_tokens = [ps.stem(word) for word in tokens if word.isalnum() and word not in stop_words]

    # Return the cleaned and transformed text
    return " ".join(processed_tokens)