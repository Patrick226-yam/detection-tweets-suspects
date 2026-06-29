import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split

# Chargement du modèle et du vectorizer
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Recharger les données avec la même correction de labels
df = pd.read_csv('data/tweets_preprocessed.csv')
df['text_clean'] = df['text_clean'].fillna('')

df['label'] = 1 - df['label']

X_raw = df['text_clean'].values
y = df['label'].values

# Même split que train.py
X = tfidf.transform(X_raw)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Prédictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Métriques
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("=== Rapport de classification ===")
print(classification_report(y_test, y_pred,
                             target_names=['Non-suspect', 'Suspect'],
                             zero_division=0))

metrics = {
    "accuracy":  round(accuracy, 4),
    "precision": round(precision, 4),
    "recall":    round(recall, 4),
    "f1_score":  round(f1, 4)
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Métriques sauvegardées :", metrics)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Non-suspect', 'Suspect'])
disp.plot(ax=ax, colorbar=True, cmap='Blues')
ax.set_title("Matrice de confusion — XGBoost")
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
plt.close()
print("Matrice de confusion sauvegardée : confusion_matrix.png")

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='steelblue', lw=2,
        label=f'XGBoost (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
ax.set_xlabel('Taux de faux positifs')
ax.set_ylabel('Taux de vrais positifs')
ax.set_title('Courbe ROC — XGBoost')
ax.legend(loc='lower right')
ax.grid(True)
plt.tight_layout()
plt.savefig('courbe_roc.png', dpi=100)
plt.close()
print(f"Courbe ROC sauvegardée (AUC = {roc_auc:.4f}) : courbe_roc.png")
