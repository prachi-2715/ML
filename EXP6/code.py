# -----------------------------------------
# EXPERIMENT 6: AdaBoost Algorithm on Adult Census Income Dataset
# -----------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

print("📂 Loading dataset...")

# === 1. Load dataset ===
try:
    df = pd.read_csv("dataset.csv")  # ensure this file is in the same folder
    print("✅ Dataset loaded successfully!")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

print(f"Shape: {df.shape}")
print("Columns:", list(df.columns))
print("\n🔹 First few rows:")
print(df.head())

# === 2. Separate target and features ===
target_col = 'income'
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if target_col in categorical_cols:
    categorical_cols.remove(target_col)  # remove income from categorical features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("\n📊 Categorical Columns:", categorical_cols)
print("🔢 Numerical Columns:", numerical_cols)

# === 3. Handle missing values ===
df = df.replace('?', np.nan)
df = df.dropna()

# === 4. Encode categorical columns ===
label_encoders = {}
for col in categorical_cols + [target_col]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === 5. Split dataset ===
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# === 6. Train AdaBoost model ===
print("\n🚀 Training AdaBoost Classifier...")
model = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, random_state=42)
model.fit(X_train, y_train)

# === 7. Evaluate model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\n✅ Model Evaluation:")
print(f"Accuracy: {acc:.3f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# === 8. Feature Importance Plot ===
feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette="viridis", legend=False)
plt.title("Feature Importance (AdaBoost Classifier)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()

# === 9. Predict for a new person ===
print("\n🔍 Predicting income for a new person...")

new_person = {
    'age': 32,
    'workclass': 'Private',
    'fnlwgt': 186824,
    'education': 'HS-grad',
    'education-num': 9,
    'marital-status': 'Married-civ-spouse',
    'occupation': 'Machine-op-inspct',
    'relationship': 'Husband',
    'race': 'White',
    'sex': 'Male',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States'
}

# Convert input to DataFrame
input_df = pd.DataFrame([new_person])

# Encode categorical columns safely (handle unseen values)
for col in categorical_cols:
    le = label_encoders[col]
    input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Align with training columns
input_df = input_df[X.columns]

# Predict
pred = model.predict(input_df)[0]
pred_label = label_encoders[target_col].inverse_transform([pred])[0]

print(f"\n💡 Predicted Income Category: {pred_label}")
