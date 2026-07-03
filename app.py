import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

message = input("Enter your message: ")

message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)
probability = model.predict_proba(message_vector)

print("Prediction:", prediction)
print("Probability:", probability)

if prediction[0] == 1:
    print("🚨 Spam Email")
else:
    print("✅ Not Spam")