"""
preprocessing.py  —  IT Helpdesk Ticket Classifier
===================================================
Provides all text-cleaning, feature-extraction, and encoding utilities
used across the project.  Import from this module in every notebook so
that preprocessing is identical between training and inference.

FUNCTIONS EXPORTED
  clean_text(text)               -> str   (lowercase, no punct, stemmed)
  build_tfidf(train, val, test)  -> (sparse, sparse, sparse, TfidfVectorizer)
  build_w2v(train, val, test)    -> (ndarray, ndarray, ndarray, Word2Vec)
  encode_labels(series, le)      -> ndarray  (fit or transform)
  make_splits(df)                -> (train_df, val_df, test_df)
"""

import re
import numpy as np
import joblib
from pathlib import Path

# ── NLTK setup ─────────────────────────────────────────────────────────────────
import nltk

def _download_nltk():
    """Download required NLTK data if not already present."""
    for pkg in ['stopwords', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
        except (LookupError, OSError):
            nltk.download(pkg, quiet=True)

_download_nltk()

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_STEMMER   = PorterStemmer()
_STOPWORDS = set(stopwords.words('english'))

# ── sklearn / gensim ──────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec


# ═══════════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Full text-cleaning pipeline applied to every ticket before feature extraction.

    Steps (in order):
      1. Lowercase                   — normalise case
      2. Remove punctuation/symbols  — keep only letters, digits, spaces
      3. Replace digit sequences     — map all numbers to the token 'NUM'
                                       (e.g. 'error 0x80070005' -> 'error NUM')
      4. Tokenise by whitespace
      5. Remove NLTK English stopwords
      6. Porter stemming             — reduces vocab size, e.g. 'connecting' -> 'connect'

    Parameters
    ----------
    text : str  — raw ticket description

    Returns
    -------
    str — cleaned, stemmed, space-joined tokens
    """
    # 1. lowercase
    text = text.lower()

    # 2. remove anything that is not a-z, 0-9, or whitespace
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # 3. replace numeric tokens with placeholder
    text = re.sub(r'\b\d+\b', 'NUM', text)

    # 4. tokenise
    tokens = text.split()

    # 5. remove stopwords
    tokens = [t for t in tokens if t not in _STOPWORDS]

    # 6. stem
    tokens = [_STEMMER.stem(t) for t in tokens if len(t) > 1]

    return ' '.join(tokens)


# ═══════════════════════════════════════════════════════════════════════════════
#  TF-IDF FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_tfidf(train_texts, val_texts, test_texts,
                max_features: int = 5000,
                ngram_range: tuple = (1, 2)):
    """
    Fit a TfidfVectorizer on training texts only, then transform all splits.
    Fits only on training data to prevent data leakage.

    Parameters
    ----------
    train_texts   : list[str]  — cleaned training ticket texts
    val_texts     : list[str]  — cleaned validation texts
    test_texts    : list[str]  — cleaned test texts
    max_features  : int        — vocabulary size cap (default 5 000)
    ngram_range   : tuple      — unigrams + bigrams (1,2) by default

    Returns
    -------
    X_train, X_val, X_test : scipy sparse matrices  (shape: n_samples x max_features)
    vectorizer             : fitted TfidfVectorizer  (save with joblib for inference)
    """
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,       # log(1 + tf) — reduces impact of very common terms
        min_df=2,                # ignore terms appearing in < 2 documents
        strip_accents='unicode',
    )

    X_train = vectorizer.fit_transform(train_texts)   # fit + transform
    X_val   = vectorizer.transform(val_texts)          # transform only
    X_test  = vectorizer.transform(test_texts)         # transform only

    print(f'TF-IDF vocabulary size : {len(vectorizer.vocabulary_):,}')
    print(f'Feature matrix shapes  : train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')

    return X_train, X_val, X_test, vectorizer


# ═══════════════════════════════════════════════════════════════════════════════
#  WORD2VEC FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def _avg_vector(tokens: list, model: Word2Vec, vector_size: int) -> np.ndarray:
    """
    Compute the average Word2Vec vector for a list of tokens.
    OOV (out-of-vocabulary) tokens are ignored.
    If ALL tokens are OOV, return a zero vector.

    This 'average-pooling' approach converts a variable-length token sequence
    into a fixed-size dense feature vector suitable for sklearn estimators.
    """
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    return np.zeros(vector_size)


def build_w2v(train_texts, val_texts, test_texts,
              vector_size: int = 100,
              window: int = 5,
              min_count: int = 2,
              seed: int = 42):
    """
    Train a Word2Vec model on training texts only, then compute averaged
    document vectors for all three splits.
    Trains only on training data to prevent data leakage.

    Parameters
    ----------
    train_texts  : list[str]  — cleaned training texts (space-joined tokens)
    val_texts    : list[str]
    test_texts   : list[str]
    vector_size  : int        — embedding dimension (100 matches common W2V sizes)
    window       : int        — context window for skip-gram/CBOW
    min_count    : int        — ignore tokens with corpus freq < min_count
    seed         : int

    Returns
    -------
    X_train_w2v, X_val_w2v, X_test_w2v : ndarray  (n_samples x vector_size)
    w2v_model                           : trained Word2Vec object
    """
    # Tokenise (split on whitespace — text is already cleaned)
    train_tok = [t.split() for t in train_texts]
    val_tok   = [t.split() for t in val_texts]
    test_tok  = [t.split() for t in test_texts]

    # Train Word2Vec on training corpus
    w2v_model = Word2Vec(
        sentences=train_tok,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=seed,
        epochs=10,
    )

    print(f'Word2Vec vocab size : {len(w2v_model.wv):,}')
    print(f'Vector size         : {vector_size}')

    # Compute averaged document vectors for all splits
    X_train = np.array([_avg_vector(t, w2v_model, vector_size) for t in train_tok])
    X_val   = np.array([_avg_vector(t, w2v_model, vector_size) for t in val_tok])
    X_test  = np.array([_avg_vector(t, w2v_model, vector_size) for t in test_tok])

    print(f'W2V matrix shapes   : train={X_train.shape}, val={X_val.shape}, test={X_test.shape}')

    return X_train, X_val, X_test, w2v_model


# ═══════════════════════════════════════════════════════════════════════════════
#  LABEL ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def fit_label_encoder(series) -> LabelEncoder:
    """Fit and return a LabelEncoder on the given Series."""
    le = LabelEncoder()
    le.fit(series)
    print(f'Label encoder classes: {list(le.classes_)}')
    return le


def encode_labels(series, le: LabelEncoder) -> np.ndarray:
    """Transform a label Series using an already-fitted LabelEncoder."""
    return le.transform(series)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN / VAL / TEST SPLIT  (70 / 15 / 15)
# ═══════════════════════════════════════════════════════════════════════════════

def make_splits(df, target_col: str = 'category', seed: int = 42):
    """
    Stratified 70 / 15 / 15 split.

    Stratification is done on `target_col` (default: category) to ensure
    each split contains proportional representation of every class.
    This is especially important for the minority classes (Database, Critical).

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    # Step 1: hold out 15 % for test, stratified
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        stratify=df[target_col],
        random_state=seed,
    )

    # Step 2: split remaining 85 % into 70 / 15 overall
    # 15 % of original  =  15/85 ≈ 17.65 % of the remaining 85 %
    train, val = train_test_split(
        train_val,
        test_size=0.1765,
        stratify=train_val[target_col],
        random_state=seed,
    )

    print(f'Split sizes  —  train: {len(train)}, val: {len(val)}, test: {len(test)}')
    print(f'Train category distribution:\n{train[target_col].value_counts()}')

    return train.reset_index(drop=True), \
           val.reset_index(drop=True),   \
           test.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SAVE / LOAD HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def save_artifact(obj, path: str):
    """Save any sklearn/joblib-serialisable object."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    print(f'Saved  : {path}')


def load_artifact(path: str):
    """Load a joblib artifact."""
    return joblib.load(path)
