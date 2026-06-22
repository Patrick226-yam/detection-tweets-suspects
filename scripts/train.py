import pickle
import os

os.makedirs('models', exist_ok=True)

# Placeholder - sera remplacé en Partie 4
model = {"status": "placeholder"}

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Train terminé !")