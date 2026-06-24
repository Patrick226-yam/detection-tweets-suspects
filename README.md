# Détection de Tweets Suspects

> Projet d'examen final — Master FIDA, Semestre 3  
> Construction de Modèles et leur Déploiement  
> Enseignant : Dr. Abdoul Kader KABORE

---

## Description

Ce projet développe une solution complète de **classification automatique de tweets suspects** en couvrant l'ensemble du cycle de vie d'un projet de Machine Learning :

- Exploration et prétraitement des données textuelles
- Représentation vectorielle (TF-IDF, Word2Vec)
- Construction et comparaison de modèles (Logistic Regression, Random Forest, XGBoost)
- Gestion du déséquilibre des classes (SMOTE)
- Pipeline reproductible avec **DVC**
- Déploiement sous forme d'application **Streamlit**

---

## 📁 Structure du projet

```
CODE/
├── data/
│   ├── tweets_suspect.csv           # Dataset brut (60 000 tweets)
│   └── tweets_preprocessed.csv     # Dataset après prétraitement
├── models/
│   ├── model.pkl                    # Modèle XGBoost entraîné
│   ├── tfidf_vectorizer.pkl         # Vectoriseur TF-IDF
│   └── word2vec.model               # Modèle Word2Vec
├── scripts/
│   ├── preprocess.py                # Script de prétraitement
│   ├── train.py                     # Script d'entraînement
│   └── evaluate.py                  # Script d'évaluation
├── app.py                           # Application Streamlit
├── EXAMEN.ipynb                     # Notebook d'analyse complet
├── dvc.yaml                         # Pipeline DVC
├── dvc.lock                         # Verrou DVC
├── metrics.json                     # Métriques du modèle
└── README.md
```

---

## Dataset

- **Source** : Dérivé du corpus Sentiment140 (tweets en anglais)
- **Taille** : 60 000 tweets
- **Classes** :
  - `0` : Non-suspect (tweets normaux) : 53 855 exemples (89,8%)
  - `1` : Suspect (tweets négatifs/offensants) : 6 145 exemples (10,2%)

> **NB** : Les labels du dataset original étaient inversés (0 = négatif, 1 = positif). Une correction a été appliquée dans les scripts (`label = 1 - label`).

---

## Installation

### Prérequis

- Python 3.8+
- Git
- DVC

### 1. Cloner le dépôt

```bash
git clone https://github.com/Patrick226-yam/detection-tweets-suspects.git
cd detection-tweets-suspects
```

### 2. Installer les dépendances

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn gensim nltk streamlit joblib matplotlib seaborn wordcloud dvc
```

### 3. Télécharger les données NLTK

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

---

## Reproduire le pipeline DVC

### Récupérer les données

```bash
dvc pull
```

### Lancer le pipeline complet

```bash
dvc repro
```

Le pipeline exécute automatiquement les 3 étapes :

```
preprocess → train → evaluate
```

### Visualiser le pipeline

```bash
dvc dag
```

---

## Lancer l'application Streamlit

```bash
streamlit run app.py
```

Ouvre ensuite `http://localhost:8501` dans ton navigateur.

---

## Résultats

| Modèle | Accuracy | F1-Score | AUC |
|---|---|---|---|
| Logistic Regression | 92% | 71% | 0.87 |
| Random Forest | 94% | 73% | 0.88 |
| **XGBoost ✓** | **95%** | **74%** | **0.89** |

### Métriques détaillées — XGBoost

| Classe | Precision | Recall | F1-Score |
|---|---|---|---|
| Non-suspect | 96% | 99% | 97% |
| Suspect | 86% | 65% | 74% |
| **Global** | **95%** | **95%** | **95%** |

---

##Technologies utilisées

| Outil | Usage |
|---|---|
| Python 3 | Langage principal |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Modèles ML et métriques |
| XGBoost | Modèle final |
| imbalanced-learn | SMOTE |
| NLTK | Prétraitement NLP |
| Gensim | Word2Vec |
| DVC | Versionnement données et pipeline |
| Git | Versionnement code |
| Streamlit | Application web |
| Matplotlib / Seaborn | Visualisations |

---

## Auteur

**YAMEOGO Wend-Gudi Pascal Patrick**  
Etudiant en Master 2 Fouille de Données et IA
Juin 2026
