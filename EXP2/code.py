# ===============================
# TITANIC SURVIVAL PREDICTION USING LOGISTIC REGRESSION
# ===============================

# ---- AIM ----
# Analyze the Titanic Survival Dataset and apply Logistic Regression
# to predict passenger survival based on given features.

# ---- IMPORT LIBRARIES ----
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---- LOAD DATASET ----
df = pd.read_csv("dataset.csv")

print("---- First 10 Records ----")
print(df.head(10))

print("\n---- Dataset Info ----")
print(df.info())

print("\n---- Missing Values ----")
print(df.isnull().sum())

# ---- HANDLE MISSING VALUES ----
df.fillna({
    'Age': df['Age'].median(),
    'Embarked': df['Embarked'].mode()[0],
    'Fare': df['Fare'].median()
}, inplace=True)

# ---- DROP UNUSED COLUMNS ----
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# ---- CONVERT CATEGORICAL VARIABLES TO NUMERIC ----
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ---- CHECK CLEANED DATA ----
print("\n---- Cleaned Dataset ----")
print(df.head())

# ---- SPLIT DATA ----
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- TRAIN MODEL ----
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ---- MAKE PREDICTIONS ----
y_pred = model.predict(X_test)

# ---- EVALUATE MODEL ----
print("\n---- Model Evaluation ----")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---- FEATURE IMPORTANCE ----
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print("\n---- Feature Importance ----")
print(importance)

# ---- VISUALIZATIONS ----
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

sns.barplot(x='Sex', y='Survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

sns.barplot(x='Pclass', y='Survived', data=df)
plt.title("Survival Rate by Passenger Class")
plt.show()

print("\n✅ Logistic Regression successfully applied on Titanic dataset.")
