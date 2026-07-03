import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page settings
st.set_page_config(
    page_title="AI Powered Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# Title
st.title("📧 AI Powered Spam Email Detector")
st.write("Enter an email below to check whether it is **Spam** or **Not Spam**.")

# Input
message = st.text_area("Enter Email", height=200)

# Button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter an email.")
    else:
        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)[0]
        probability = model.predict_proba(message_vector)[0]
        confidence = max(probability) * 100

        if prediction == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam")

        st.info(f"Confidence: {confidence:.2f}%")

# Footer
st.markdown("---")
st.caption("Developed by Kunal | Python • Scikit-learn • Logistic Regression • TF-IDF")