# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load dataset
df = pd.read_csv('disease.csv')

# 2. Separate features (X) and target (y)
X = df.drop('prognosis', axis=1)
y = df['prognosis']

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# 5. Evaluate model performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 6. Save model for later use
pickle.dump(model, open('model.pkl', 'wb'))
print("\n💾 Model saved as model.pkl")
