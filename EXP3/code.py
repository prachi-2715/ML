# -------------------------------
# EXP3 – Decision Tree on Adult Census Income Dataset
# Aim: Apply Decision Tree Algorithm and analyze performance
# -------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Load Dataset ----
print("---- Loading Dataset ----")

try:
    df = pd.read_csv("dataset.csv")
except FileNotFoundError:
    raise FileNotFoundError("⚠️ dataset.csv not found. Please place it in the same folder as code.py")

# Clean column names
df.columns = [c.strip().lower().replace(" ", "-") for c in df.columns]

# Check and rename target if necessary
if 'income' not in df.columns:
    for c in df.columns:
        if 'income' in c:
            df.rename(columns={c: 'income'}, inplace=True)

print(df.head(), "\n")

# ---- Handle Missing Values ----
print("---- Missing Values Before Cleaning ----")
print(df.isnull().sum())

df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

print("\n---- Missing Values After Cleaning ----")
print(df.isnull().sum())

# ---- Label Encoding ----
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

print("\n---- After Encoding ----")
print(df.head(), "\n")

# ---- Split Features & Target ----
X = df.drop('income', axis=1)
y = df['income']

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---- Model Training with GridSearchCV ----
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 9, 12, 15],
    'min_samples_split': [2, 5, 10, 20, 50]
}

dt = DecisionTreeClassifier(random_state=42, class_weight='balanced')
grid = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("\n---- Best Parameters ----")
print(grid.best_params_)
print(f"Best Cross-Validation Accuracy: {grid.best_score_:.4f}\n")

# ---- Best Model ----
best_dt = grid.best_estimator_
y_pred = best_dt.predict(X_test)

# ---- Evaluation ----
print("---- Model Evaluation ----")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}\n")

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm, "\n")

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---- Classification Report ----
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# ---- Feature Importance ----
importances = pd.Series(best_dt.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index, palette='viridis')
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# ---- Visualize Decision Tree ----
plt.figure(figsize=(20, 10))
plot_tree(best_dt, filled=True, feature_names=X.columns, class_names=['<=50K', '>50K'], rounded=True, fontsize=8)
plt.title("Decision Tree Visualization")
plt.show()
