# ---- Boston Housing Linear Regression (handles '|' separated dataset) ----

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ---- Step 1: Load dataset properly ----
# Your dataset likely uses '|' as a separator and has extra spaces
df = pd.read_csv("dataset.csv", sep='|', engine='python')

# Remove empty columns and strip whitespace
df = df.dropna(axis=1, how='all')
df.columns = [c.strip() for c in df.columns]
df = df.apply(lambda x: x.astype(str).str.strip())

# Drop rows that are headers or lines
df = df[df['CRIM'] != 'CRIM']  # remove header rows that may repeat
df = df[df['CRIM'] != '-------']  # remove separator rows

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Drop NaN rows
df.dropna(inplace=True)

print("---- Cleaned Dataset (First 10 Records) ----")
print(df.head(10))
print("\nDataset Shape:", df.shape)

# ---- Step 2: Correlation heatmap ----
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of Boston Housing Dataset")
plt.show()

# ---- Step 3: Split features and target ----
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# ---- Step 4: Scale features ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Step 5: Train-test split ----
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ---- Step 6: Train Linear Regression model ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- Step 7: Predict and evaluate ----
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n---- Model Evaluation ----")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# ---- Step 8: Compare actual vs predicted ----
comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\n---- Actual vs Predicted ----")
print(comparison.head(10))

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted Median Home Value")
plt.show()
