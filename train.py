"""import pandas as pd

# Load the dataset
df = pd.read_csv("email.csv", encoding="latin-1")

# Show first 5 rows
print(df.head())

# Show column names
print(df.columns)"""

import joblib

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("email.csv", encoding="latin-1")

# Rest of your code...

import pandas as pd

# Load dataset
df = pd.read_csv("email.csv", encoding="latin-1")

# Dataset information
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Count spam and ham emails
print("\nCategory Count:")
print(df["Category"].value_counts())


# Convert ham and spam into numbers
# Remove extra spaces and convert to lowercase
df["Category"] = df["Category"].str.strip().str.lower()

# Convert ham and spam into numbers
df["Category"] = df["Category"].map({
    "ham": 0,
    "spam": 1
})

# Remove rows where Category became NaN
df = df.dropna(subset=["Category"])

# Convert Category to integer
df["Category"] = df["Category"].astype(int)

# Features
X = df["Message"]

# Labels
y = df["Category"]

# Check if mapping worked correctly
print(df["Category"].unique())
print("Missing labels:", df["Category"].isnull().sum())

# Show rows where Category is NaN
print(df[df["Category"].isnull()])

print("\nAfter Encoding:")
print(df.head())


# X = Messages
X = df["Message"]

# y = Category (0 or 1)
y = df["Category"]

print("\nFirst 5 Messages:")
print(X.head())

print("\nFirst 5 Labels:")
print(y.head())




# Convert text into numerical features
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

print("\nShape of TF-IDF Matrix:")
print(X.shape)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Create the model
model = MultinomialNB()

print("Unique values:", df["Category"].unique())
print("Missing values:", df["Category"].isnull().sum())
print(df["Category"].head(10))

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_train contains NaN:", X_train.data.any() if hasattr(X_train, "data") else "N/A")
print("y_train NaN count:", y_train.isnull().sum())

print(type(X_train))
print(type(y_train))
print(y_train.dtype)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

# Save the trained model
joblib.dump(model, "spam_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")