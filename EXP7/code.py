# -------------------------------
# code.py — Random Forest on Adult Census Income Dataset
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------
# 1️⃣ Load dataset (use only first 500 rows for quick testing)
# ----------------------------------------------------------
data = pd.read_csv('dataset.csv')
data = data.head(500)   # adjust here (e.g., 100, 500, or full dataset)

print("✅ Dataset loaded successfully!")
print(f"Shape: {data.shape}")
print(data.head())

# ----------------------------------------------------------
# 2️⃣ Basic Cleaning
# ----------------------------------------------------------
data = data.replace('?', np.nan)
data = data.dropna()

# Target column (Income)
y = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Feature columns
X = data.drop('income', axis=1)

# ----------------------------------------------------------
# 3️⃣ Identify categorical & numerical columns
# ----------------------------------------------------------
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(exclude=['object']).columns

# ----------------------------------------------------------
# 4️⃣ Preprocessing: OneHotEncode categorical, scale numeric
# ----------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# ----------------------------------------------------------
# 5️⃣ Dimensionality Reduction using PCA (optional visualization)
# ----------------------------------------------------------
pca = PCA(n_components=2, random_state=42)

# ----------------------------------------------------------
# 6️⃣ Random Forest Classifier
# ----------------------------------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)

# ----------------------------------------------------------
# 7️⃣ Create Pipeline
# ----------------------------------------------------------
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('pca', pca),
    ('classifier', model)
])

# ----------------------------------------------------------
# 8️⃣ Train-Test Split
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------
# 9️⃣ Train Model
# ----------------------------------------------------------
clf.fit(X_train, y_train)

# ----------------------------------------------------------
# 🔟 Evaluate Model
# ----------------------------------------------------------
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Model Performance Metrics:")
print(f"✅ Accuracy : {accuracy:.4f}")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall   : {recall:.4f}")
print(f"✅ F1 Score : {f1:.4f}")

print("\nDetailed Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------------------
# 🔹 Confusion Matrix Visualization
# ----------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ----------------------------------------------------------
# 🔮 PREDICT FOR NEW INPUT DATA
# ----------------------------------------------------------
print("\n💡 Predict Income for a New Person")

new_data = {
    'age': [32],
    'workclass': ['Private'],
    'fnlwgt': [200000],
    'education': ['Masters'],
    'education-num': [14],
    'marital-status': ['Never-married'],
    'occupation': ['Tech-support'],
    'relationship': ['Not-in-family'],
    'race': ['White'],
    'sex': ['Male'],
    'capital-gain': [5000],
    'capital-loss': [0],
    'hours-per-week': [50],
    'native-country': ['United-States']
}

new_df = pd.DataFrame(new_data)

# Use the same preprocessing and model pipeline
predicted_class = clf.predict(new_df)[0]
predicted_label = ">50K" if predicted_class == 1 else "<=50K"

print(f"\n🧠 Predicted Income Category: {predicted_label}")
