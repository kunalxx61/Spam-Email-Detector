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
# Sample Emails
# ----------------------------
spam_sample = """Congratulations! You have won ₹50,000. Click here to claim your prize immediately."""

safe_sample = """Hi Kunal, can we meet tomorrow at 2 PM to discuss our AI project?"""

# Buttons for sample emails
col1, col2 = st.columns(2)

if col1.button("🚨 Try Spam Example"):
    st.session_state.message = spam_sample

if col2.button("✅ Try Safe Example"):
    st.session_state.message = safe_sample

# Store text
if "message" not in st.session_state:
    st.session_state.message = ""

message = st.text_area(
    "Enter Email",
    value=st.session_state.message,
    height=200
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter an email.")
    else:

        with st.spinner("🔍 Analyzing Email..."):

            message_vector = vectorizer.transform([message])

            prediction = model.predict(message_vector)[0]
            probability = model.predict_proba(message_vector)[0]

            confidence = max(probability) * 100

        st.success("Analysis Completed ✅")

        if prediction == 1:
            st.error("🚨 Spam Email Detected")
        else:
            st.success("✅ This Email Looks Safe")

        st.progress(int(confidence))

        st.info(f"Confidence: {confidence:.2f}%")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Developed by Kunal | Python • Scikit-learn • Logistic Regression • TF-IDF")