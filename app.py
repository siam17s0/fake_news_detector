import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# NLTK download (first time)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith('J'): return wordnet.ADJ
    elif tag.startswith('V'): return wordnet.VERB
    elif tag.startswith('N'): return wordnet.NOUN
    elif tag.startswith('R'): return wordnet.ADV
    else: return wordnet.NOUN

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    final_words = []
    for word, tag in tags:
        if word not in stop_words and word.isalpha():
            pos = get_wordnet_pos(tag)
            lemma = lemmatizer.lemmatize(word, pos)
            final_words.append(lemma)
    return " ".join(final_words)

# Load TF-IDF & LinearSVC
tfidf = pickle.load(open("tfidf.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

def predict_news(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    pred = model.predict(vectorized)[0]
    return pred

# UI Design

st.set_page_config(page_title="Fake News AI", layout="centered")

st.markdown("<h1 style='text-align: center;'>🧠 Fake News AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask anything about a news article...</p>", unsafe_allow_html=True)

# Input (Enter press works automatically)
user_input = st.text_input(
    "",
    placeholder="Enter news text here... (Press Enter to predict)"
)

# Prediction auto trigger (ChatGPT style)
if user_input:
    pred = predict_news(user_input)

    st.markdown("---")

    if pred == 0:
        st.error("🚨 FAKE news Detected")
    else:
        st.success("✅ REAL news")

# Footer
st.markdown("---")
st.caption("⚡ Powered by TF-IDF + LinearSVC")