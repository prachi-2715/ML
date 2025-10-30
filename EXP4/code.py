# code.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("📂 Loading dataset...")
df = pd.read_csv("dataset.csv")

print("✅ Dataset loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\n🔹 First few rows:")
print(df.head(), "\n")

# --- Step 1: Identify categorical and numerical columns ---
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(exclude=['object']).columns.tolist()
print("📊 Categorical Columns:", categorical_cols)
print("🔢 Numerical Columns:", numerical_cols, "\n")

# --- Step 2: Encode categorical variables ---
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- Step 3: Define features (X) and target (y) ---
# The target column in the dataset is 'income'
X = df.drop('income', axis=1)
y = df['income']

# --- Step 4: Split dataset (80% train, 20% test) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Training the Random Forest model...\n")

# --- Step 5: Train Random Forest model ---
model = RandomForestClassifier(
    n_estimators=100,    # number of trees
    max_depth=None,      # let the trees expand fully
    random_state=42
)
model.fit(X_train, y_train)

# --- Step 6: Predictions ---
y_pred = model.predict(X_test)

# --- Step 7: Evaluation ---
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("✅ Model Evaluation:")
print(f"Accuracy: {acc:.3f}\n")
print("Confusion Matrix:\n", cm, "\n")
print("Classification Report:\n", cr)