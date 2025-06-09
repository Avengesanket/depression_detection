import re
import string
import json
import pickle
import numpy as np
import contractions
import emoji
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter

# —————————————————————————————————————————————
# 1. Load all of your saved artifacts at module import
# —————————————————————————————————————————————
nltk.download('vader_lexicon')

# a) lexicons & sentiment dictionary
with open('depression_lexicons.json', 'r') as f:
    _lexicons = json.load(f)
sentiment_dict = _lexicons['sentiment_dict']

# b) preprocessing config
with open('preprocessing_config.json', 'r') as f:
    _config = json.load(f)
MAX_LEN      = _config['max_len']
EMBED_DIM    = _config['embedding_dim']

# c) vocabulary
with open('vocabulary.pkl', 'rb') as f:
    VOCAB = pickle.load(f)

# d) NLTK / VADER setup
STOPWORDS   = set(stopwords.words('english'))
LEMMATIZER  = WordNetLemmatizer()
STEMMER     = PorterStemmer()
TOKENIZER   = TweetTokenizer(preserve_case=False, strip_handles=False, reduce_len=True)
SID         = SentimentIntensityAnalyzer()

# —————————————————————————————————————————————
# 2. Exactly your cleaning & feature functions
# —————————————————————————————————————————————

def expand_contractions(text):
    return contractions.fix(text)

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', ' ', text)

def remove_html_tags(text):
    return re.sub(r'<.*?>', ' ', text)

def extract_hashtags(text):
    return re.findall(r'#(\w+)', text)

def extract_mentions(text):
    return re.findall(r'@(\w+)', text)

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

def replace_emojis(text):
    return emoji.demojize(text)

def remove_punctuation(text):
    text = re.sub(r'(\w)\'(\w)', r'\1\2', text)
    return re.sub(r'[^\w\s]', ' ', text)

def remove_extra_whitespace(text):
    return re.sub(r'\s+', ' ', text).strip()

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def normalize_elongated_words(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)
def preprocess_text(text, remove_stop=True, lemmatize=True, stem=False):
    """Complete preprocessing pipeline"""
    if not isinstance(text, str) or text.strip() == '':
        return []

    # Extract features before cleaning
    hashtags = extract_hashtags(text)
    mentions = extract_mentions(text)
    emojis = extract_emojis(text)

    # Store original text for sentiment analysis
    original_text = text

    # Basic cleaning
    text = text.lower()
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = replace_emojis(text)
    text = normalize_elongated_words(text)

    # Handle Reddit-specific formatting
    text = re.sub(r'\[removed\]|\[deleted\]', '', text)
    text = re.sub(r'\(.*?\)', ' ', text)  

    # Handle Twitter-specific formatting 
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'@\w+', '', text)     

    # Handle special characters and numbers
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_extra_whitespace(text)

    # Tokenize
    tokens = TOKENIZER.tokenize(text)

    # Remove stopwords
    if remove_stop:
        tokens = [token for token in tokens if token not in STOPWORDS and len(token) > 1]

    # Lemmatize
    if lemmatize:
        tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    # Stem
    if stem:
        tokens = [STEMMER.stem(token) for token in tokens]

    # Return tokens along with extracted features
    feature_dict = {
        'tokens': tokens,
        'hashtags': hashtags,
        'mentions': mentions,
        'emojis': emojis,
        'original_text': original_text,
        'token_count': len(tokens)
    }

    return feature_dict

def calculate_sentiment_features(tokens):
    total = len(tokens)
    depression_cnt = sum(1 for w in tokens if sentiment_dict.get(w,0)<0)
    positive_cnt   = sum(1 for w in tokens if sentiment_dict.get(w,0)>0)
    score          = sum(sentiment_dict.get(w,0) for w in tokens)/total if total else 0
    return np.array([ score,
                      depression_cnt/total if total else 0,
                      positive_cnt/total if total else 0,
                      depression_cnt/total if total else 0 ])

def calculate_vader_sentiment(text):
    s = SID.polarity_scores(text)
    return np.array([s['neg'], s['neu'], s['pos'], s['compound']])

def calculate_text_statistics(tokens, original_text):
    total = len(tokens)
    if total == 0:
        return np.zeros(5)
    char_count = len(original_text)
    word_count = total
    avg_len    = sum(len(t) for t in tokens)/total
    uniq       = len(set(tokens))
    lex_div    = uniq/total
    most_comm  = Counter(tokens).most_common(1)[0][1]
    rep_ratio  = most_comm/total
    return np.array([char_count, word_count, avg_len, lex_div, rep_ratio])

def calculate_social_media_features(feats):
    return np.array([len(feats['hashtags']),
                     len(feats['mentions']),
                     len(feats['emojis'])])

# —————————————————————————————————————————————
# 3. A single “transform” function for new text
# —————————————————————————————————————————————
def transform_single(text):
    """
    Given a raw `text` string returns:
      - token_indices: a length‐MAX_LEN numpy array of vocab indices
      - feature_vector: a numpy array of length 4+4+5+3 = 16
    """
    feat_dict = preprocess_text(text)

    sent_feats  = calculate_sentiment_features(feat_dict['tokens'])
    vader_feats = calculate_vader_sentiment(feat_dict['original_text'])
    stats_feats = calculate_text_statistics(feat_dict['tokens'], feat_dict['original_text'])
    social_feats= calculate_social_media_features(feat_dict)
    feature_vector = np.concatenate([sent_feats, vader_feats, stats_feats, social_feats])

    idxs = [VOCAB.get(w, VOCAB['<UNK>']) for w in feat_dict['tokens']]
    if len(idxs) < MAX_LEN:
        idxs += [VOCAB['<PAD>']] * (MAX_LEN - len(idxs))
    else:
        idxs = idxs[:MAX_LEN]

    return np.array(idxs, dtype=np.int64), feature_vector.astype(np.float32)
