# ---------------------------------------------------------------
# code.py
# Apply Unsupervised Learning (K-Means Clustering) 
# on the Wholesale Customers Dataset
# Includes interactive user prediction
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import joblib
import json

# ---------------------------------------------------------------
# STEP 1: CREATE SAMPLE DATASET (10 ROWS)
# ---------------------------------------------------------------
CSV_CONTENT = """CustomerID,Fresh,Milk,Grocery,Frozen,Detergents_Paper,Delicatessen,Channel,Region
1,12669,9656,7561,214,2674,1338,Retail,Lisbon
2,7057,9810,9568,1762,3293,1776,Hotel,Oporto
3,6353,8808,7684,2405,3516,784,Retail,Other
4,13265,1196,4221,6404,507,178,Hotel,Lisbon
5,22615,5410,7198,3915,1777,518,Retail,Oporto
6,9413,8259,512,963,755,495,Hotel,Other
7,12126,4347,1524,230,239,156,Retail,Lisbon
8,11093,13548,7507,124,2594,3882,Hotel,Oporto
9,24164,1821,6207,4243,331,208,Retail,Other
10,5176,11178,4459,226,722,268,Hotel,Lisbon
"""

with open("dataset.csv", "w") as f:
    f.write(CSV_CONTENT.strip())

print("✅ dataset.csv created successfully!")

# ---------------------------------------------------------------
# STEP 2: LOAD DATASET
# ---------------------------------------------------------------
df = pd.read_csv("dataset.csv")
feature_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicatessen"]
X = df[feature_cols].values

# ---------------------------------------------------------------
# STEP 3: DETERMINE OPTIMAL CLUSTERS USING SILHOUETTE SCORE
# ---------------------------------------------------------------
best_k = 2
best_score = -1
scores = {}

for k in range(2, 6):
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores[k] = score
    if score > best_score:
        best_k = k
        best_score = score

print("\n📊 Silhouette Scores:", scores)
print("✅ Best number of clusters (k):", best_k)

# ---------------------------------------------------------------
# STEP 4: TRAIN FINAL MODEL USING PIPELINE
# ---------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=best_k, random_state=42, n_init=20))
])

pipeline.fit(X)
df["Cluster"] = pipeline.named_steps["kmeans"].labels_

print("\n📦 Cluster counts:")
print(df["Cluster"].value_counts())

# ---------------------------------------------------------------
# STEP 5: DISPLAY CLUSTER CENTERS AND INTERPRETATION
# ---------------------------------------------------------------
centers_scaled = pipeline.named_steps["kmeans"].cluster_centers_
centers = pipeline.named_steps["scaler"].inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers, columns=feature_cols)
print("\n🏷 Cluster Centers (Original Scale):")
print(centers_df)

# Determine top 2 spending categories per cluster
interpretations = {}
for i, row in centers_df.iterrows():
    top2 = row.sort_values(ascending=False).head(2).index.tolist()
    interpretations[i] = top2

print("\n🔍 Cluster Interpretations:")
for k, v in interpretations.items():
    print(f"Cluster {k}: Top spending on {v}")

# ---------------------------------------------------------------
# STEP 6: SAVE MODEL PIPELINE
# ---------------------------------------------------------------
joblib.dump(pipeline, "kmeans_pipeline.pkl")
print("\n💾 Model saved as kmeans_pipeline.pkl")

# ---------------------------------------------------------------
# STEP 7: PREDICTION FUNCTION
# ---------------------------------------------------------------
def predict_new(sample):
    """
    Predict cluster for a new customer sample.
    sample: dict with keys matching feature_cols
    """
    if isinstance(sample, dict):
        x = np.array([sample[c] for c in feature_cols]).reshape(1, -1)
    else:
        x = np.array(sample).reshape(1, -1)

    pred = pipeline.predict(x)[0]
    scaled_x = pipeline.named_steps["scaler"].transform(x)
    centers_scaled = pipeline.named_steps["kmeans"].cluster_centers_
    dists = np.linalg.norm(centers_scaled - scaled_x, axis=1)

    return {
        "predicted_cluster": int(pred),
        "distances_to_clusters": dists.tolist(),
        "interpretation": interpretations[int(pred)]
    }

# ---------------------------------------------------------------
# STEP 8: EXAMPLE PREDICTION
# ---------------------------------------------------------------
example_new = {
    "Fresh": 10000,
    "Milk": 4000,
    "Grocery": 3000,
    "Frozen": 500,
    "Detergents_Paper": 800,
    "Delicatessen": 300
}

result = predict_new(example_new)
print("\n🧮 Example Prediction:")
print("Input Data:", example_new)
print("Predicted Cluster:", result["predicted_cluster"])
print("Cluster Interpretation:", result["interpretation"])

# Save example to JSON
with open("example_new.json", "w") as f:
    json.dump(example_new, f, indent=2)

# ---------------------------------------------------------------
# STEP 9: INTERACTIVE USER INPUT
# ---------------------------------------------------------------
print("\n✨ Now enter your own customer data for prediction:")
try:
    user_sample = {}
    for feature in feature_cols:
        val = float(input(f"Enter {feature}: "))
        user_sample[feature] = val

    user_result = predict_new(user_sample)
    print("\n🔮 Prediction Result:")
    print("Your Input:", user_sample)
    print("Predicted Cluster:", user_result["predicted_cluster"])
    print("Cluster Interpretation:", user_result["interpretation"])

except Exception as e:
    print("⚠️ Error during user input:", e)

print("\n✅ Files created: dataset.csv, kmeans_pipeline.pkl, example_new.json")
