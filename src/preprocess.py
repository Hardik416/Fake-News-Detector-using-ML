"""
preprocess.py
-------------
Text cleaning and TF-IDF vectorization pipeline.
Works for both English and translated Multilingual text.
"""

import re
import string
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Optional: nltk stopwords (install once)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    NLTK_AVAILABLE = True
except ImportError:
    STOPWORDS = set()
    NLTK_AVAILABLE = False


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    1. Lowercase
    2. Remove URLs, mentions, hashtags
    3. Remove punctuation and digits
    4. Remove extra whitespace
    5. Remove stopwords (if NLTK available)
    6. Stem words (optional, helps for classical ML)
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove stopwords and apply stemming
    if NLTK_AVAILABLE and len(text) > 0:
        tokens = text.split()
        tokens = [stemmer.stem(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
        text = ' '.join(tokens)

    return text


def clean_dataframe(df: pd.DataFrame, text_col: str = 'text', label_col: str = 'label') -> pd.DataFrame:
    """Clean a dataframe of news articles."""
    df = df.copy()
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 10]  # drop very short articles

    # Normalize labels to 0 (real) and 1 (fake)
    if label_col in df.columns:
        if df[label_col].dtype == object:
            label_map = {'real': 0, 'true': 0, 'REAL': 0, 'TRUE': 0,
                         'fake': 1, 'false': 1, 'FAKE': 1, 'FALSE': 1}
            df[label_col] = df[label_col].map(label_map).fillna(df[label_col])
        df[label_col] = df[label_col].astype(int)

    return df


# ─── Dataset Loaders ─────────────────────────────────────────────────────────

def load_welfake(path: str = 'datasets/welfake_dataset.csv') -> pd.DataFrame:
    """Load and clean WELFake dataset (72K articles)."""
    df = pd.read_csv(path)
    # WELFake columns: Unnamed:0, title, text, label (0=real, 1=fake)
    df = df[['title', 'text', 'label']].copy()
    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = clean_dataframe(df, text_col='text', label_col='label')
    return df[['text', 'label']]


def load_isot(fake_path: str = 'datasets/isot_fake.csv',
              real_path: str = 'datasets/isot_real.csv') -> pd.DataFrame:
    """Load and clean ISOT Fake News dataset (44K articles)."""
    fake = pd.read_csv(fake_path)
    real = pd.read_csv(real_path)
    fake['label'] = 1
    real['label'] = 0
    df = pd.concat([fake, real], ignore_index=True)
    df['text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df = clean_dataframe(df, text_col='text', label_col='label')
    return df[['text', 'label', 'subject']] if 'subject' in df.columns else df[['text', 'label']]


# ─── TF-IDF Vectorizer ────────────────────────────────────────────────────────

def build_tfidf(texts, max_features: int = 50000, ngram_range: tuple = (1, 2)):
    """Fit a TF-IDF vectorizer on training texts."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,        # log normalization
        min_df=3,                 # ignore very rare terms
        max_df=0.95               # ignore terms in >95% of docs
    )
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Split cleaned DataFrame into train/test sets."""
    X = df['text'].values
    y = df['label'].values
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def save_vectorizer(vectorizer, path: str = 'models/tfidf_vectorizer.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved → {path}")


def load_vectorizer(path: str = 'models/tfidf_vectorizer.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)

