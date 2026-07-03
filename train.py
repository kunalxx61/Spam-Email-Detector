import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv("combined_data.csv")

# ==========================================
# Dataset Information
# ==========================================

print("\nDataset Information:")
print(df.info())

# Remove missing values
df = df.dropna()

# Remove duplicate emails
df = df.drop_duplicates()

# ==========================================
# Check Dataset
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSpam / Ham Count:")
print(df["label"].value_counts())

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# Features and Labels
# ==========================================

X = df["text"]
y = df["label"]

print("\nFirst 5 Emails:")
print(X.head())

print("\nFirst 5 Labels:")
print(y.head())

# ==========================================
# Convert Text into TF-IDF Features
# ==========================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2
)

X = vectorizer.fit_transform(X)

print("\nTF-IDF Matrix Shape:")
print(X.shape)

# ==========================================
# Split Dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================
# Create Model
# ==========================================

model = LogisticRegression(max_iter=1000)

# ==========================================
# Train Model
# ==========================================

print("\nTraining model...")

model.fit(X_train, y_train)

# ==========================================
# Predict
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy: {:.2f}%".format(accuracy * 100))

# ==========================================
# Save Model
# ==========================================

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel and Vectorizer saved successfully!")