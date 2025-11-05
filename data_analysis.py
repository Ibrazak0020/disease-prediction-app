import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
data_path = Path("disease.csv")

if not data_path.exists():
    raise FileNotFoundError(f"❌ Cannot find {data_path.resolve()}")

df = pd.read_csv(data_path)
print(f"✅ Loaded dataset with shape: {df.shape}")

# --------------------------------------------------
# 2. Basic info
# --------------------------------------------------
print("\n📋 Columns:")
print(df.columns.tolist())

print("\n📊 Number of unique diseases (target classes):", df['prognosis'].nunique())

# --------------------------------------------------
# 3. Check duplicates
# --------------------------------------------------
duplicates = df.duplicated().sum()
print(f"\n🔁 Duplicate rows: {duplicates}")

# --------------------------------------------------
# 4. Check if each disease has unique symptom pattern
# --------------------------------------------------
# Remove target column for uniqueness check
symptom_patterns = df.drop('prognosis', axis=1)
unique_patterns = symptom_patterns.drop_duplicates().shape[0]

if unique_patterns == df.shape[0]:
    print("🧩 Each row has a unique symptom pattern — likely deterministic.")
else:
    print("✅ Some symptom patterns repeat — more realistic dataset.")

# --------------------------------------------------
# 5. Correlation check (to detect if any symptom uniquely identifies a disease)
# --------------------------------------------------
# Convert target to numeric for correlation
df_encoded = df.copy()
df_encoded['prognosis'] = df_encoded['prognosis'].astype('category').cat.codes

corr = df_encoded.corr()['prognosis'].sort_values(ascending=False)
print("\n🔥 Top 10 symptoms most correlated with disease:\n")
print(corr.head(10))

# --------------------------------------------------
# 6. Optional visualization — correlation heatmap
# --------------------------------------------------
plt.figure(figsize=(10, 6))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', cbar=False)
plt.title("Correlation Heatmap (Prognosis vs Symptoms)")
plt.show()
