import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

os.makedirs('models', exist_ok=True)

# Chargement des données prétraitées
df = pd.read_csv('data/tweets_preprocessed.csv')
df['text_clean'] = df['text_clean'].fillna('')

# ✅ Correction des labels inversés : 0=suspect, 1=normal → on inverse
df['label'] = 1 - df['label']
# Maintenant : 1 = Suspect (6145), 0 = Non-suspect (53855)

X_raw = df['text_clean'].values
y = df['label'].values

print("Shape:", X_raw.shape)
print("Distribution classes après correction:")
print("  Non-suspect (0):", (y == 0).sum())
print("  Suspect     (1):", (y == 1).sum())

# Vectorisation TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(X_raw)

# Sauvegarde du vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Split train/test stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# Gestion du déséquilibre avec SMOTE
print("Application de SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("Après SMOTE:", dict(zip(*np.unique(y_train_bal, return_counts=True))))

# Entraînement XGBoost
print("Entraînement XGBoost...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_bal, y_train_bal)

# Vérification rapide
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"F1-Score sur test : {f1:.4f}")

# Sauvegarde du modèle
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Train terminé ! Modèle sauvegardé dans models/model.pkl")
