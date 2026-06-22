import json

# Placeholder - sera remplacé en Partie 6
metrics = {
    "accuracy": 0.0,
    "precision": 0.0,
    "recall": 0.0,
    "f1_score": 0.0
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)

print("✅ Evaluate terminé !")