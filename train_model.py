# train_model.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# --------------------------------------------------
# 1. Load dataset safely
# --------------------------------------------------
data_path = Path("disease.csv")

if not data_path.exists():
    raise FileNotFoundError(f"❌ Dataset not found at: {data_path.resolve()}")

df = pd.read_csv(data_path)
print(f"✅ Loaded dataset with shape: {df.shape}")

# --------------------------------------------------
# 2. Add controlled noise (reduce 100% accuracy)
# --------------------------------------------------
symptom_cols = df.columns.drop('prognosis')

# Randomly flip ~3% of symptom values (1 ↔ 0)
for col in symptom_cols:
    mask = np.random.rand(len(df)) < 0.03
    df.loc[mask, col] = 1 - df.loc[mask, col]

print("🔄 Added 3% random noise to symptom columns.")

# --------------------------------------------------
# 3. Split features and target
# --------------------------------------------------
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# --------------------------------------------------
# 4. Split into training/testing sets
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"📊 Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# --------------------------------------------------
# 5. Train Random Forest model
# --------------------------------------------------
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,        # restrict depth to avoid overfitting
    random_state=42,
    n_jobs=-1
)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"\n🔍 Cross-validation accuracy (5-fold): {cv_scores.mean():.2f}")

# Fit on training data
model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Evaluate model performance
# --------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Test Set Accuracy: {accuracy:.2f}\n")

print("📄 Classification Report:\n")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 7. Confusion Matrix Visualization
# --------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(14, 12))
sns.heatmap(cm_normalized, annot=False, cmap='Blues')
plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 8. Save trained model
# --------------------------------------------------
model_path = Path("model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n💾 Model saved successfully as: {model_path.resolve()}")
