import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 AI Powered Spam Email Detector")
st.write("Enter an email or SMS message below to check whether it is Spam or Not Spam.")

message = st.text_area("Enter Message")

if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)
        probability = model.predict_proba(message_vector)
        confidence = max(probability[0]) * 100

        if prediction[0] == 1:
            st.error(f"🚨 Spam Email\n\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ Not Spam\n\nConfidence: {confidence:.2f}%")