import streamlit as st
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Powered Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

# ----------------------------
# Cache Model
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# Title
# ----------------------------
st.title("📧 AI Powered Spam Email Detector")
st.write("Enter an email below to check whether it is **Spam** or **Not Spam**.")

# ----------------------------
# Input
# ----------------------------
message = st.text_area("Enter Email", height=200)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter an email.")
    else:
        with st.spinner("Analyzing email..."):

            message_vector = vectorizer.transform([message])

            prediction = model.predict(message_vector)[0]
            probability = model.predict_proba(message_vector)[0]
            confidence = max(probability) * 100

        if prediction == 1:
            st.error("🚨 Spam Email")
        else:
            st.success("✅ Not Spam")

        st.progress(int(confidence))
        st.info(f"Confidence: {confidence:.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed by Kunal | Python • Scikit-learn • Logistic Regression • TF-IDF")