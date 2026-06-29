import streamlit as st
import joblib
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Chargement du modèle et du vectoriseur (chemins relatifs)
@st.cache_resource
def load_model():
    base = os.path.dirname(os.path.abspath(__file__))
    tfidf = joblib.load(os.path.join(base, 'models', 'tfidf_vectorizer.pkl'))
    model = joblib.load(os.path.join(base, 'models', 'model.pkl'))
    return tfidf, model

tfidf, model = load_model()

# Prétraitement identique à l'entraînement
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# Interface Streamlit
st.set_page_config(page_title="Détection de Tweet Suspect", page_icon="")
st.title("Détection de Tweet Suspect")
st.markdown("Entrez un tweet pour savoir s'il est **suspect** ou **non suspect**.")

tweet = st.text_area(" Saisir le tweet ici :", height=150)

if st.button("Analyser"):
    if tweet.strip() == "":
        st.warning("Veuillez entrer un tweet.")
    else:
        processed = preprocess(tweet)
        vect = tfidf.transform([processed])
        probability = model.predict_proba(vect)[0]

        # Labels corrigés : 1 = Suspect, 0 = Non-suspect
        # Seuil ajusté à 0.3 car la classe suspecte est minoritaire (10%)
        proba_suspect = probability[1]
        prediction = 1 if proba_suspect >= 0.3 else 0

        st.markdown("---")

        if prediction == 1:
            st.error("Tweet **SUSPECT** détecté !")
        else:
            st.success("Tweet **NON SUSPECT**")

        st.markdown("### Probabilités")
        col1, col2 = st.columns(2)
        col1.metric("Non Suspect", f"{probability[0]*100:.1f}%")
        col2.metric("Suspect",     f"{probability[1]*100:.1f}%")

        with st.expander("Texte après prétraitement"):
            st.write(processed if processed else "(vide après nettoyage)")
