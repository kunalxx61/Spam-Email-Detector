import tkinter as tk
from tkinter import messagebox
import joblib

# ==========================
# Load Model and Vectorizer
# ==========================
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# ==========================
# Function to Check Message
# ==========================
def check_message():
    message = text_box.get("1.0", tk.END).strip()

    if message == "":
        messagebox.showwarning("Warning", "Please enter a message!")
        return

    # Convert message into TF-IDF
    message_vector = vectorizer.transform([message])
  

    # Predict
    prediction = model.predict(message_vector)
    print("Message:", message)
    print("Prediction:", prediction)
    print("Probability:", model.predict_proba(message_vector))

    # Prediction confidence
    probability = model.predict_proba(message_vector)
    confidence = max(probability[0]) * 100

    if prediction[0] == 1:
        result_label.config(
            text=f"🚨 Spam Email\nConfidence: {confidence:.2f}%",
            fg="red"
        )
    else:
        result_label.config(
            text=f"✅ Not Spam\nConfidence: {confidence:.2f}%",
            fg="green"
        )


# ==========================
# Clear Text
# ==========================
def clear_text():
    text_box.delete("1.0", tk.END)
    result_label.config(text="")


# ==========================
# Main Window
# ==========================
root = tk.Tk()

root.title("AI Powered Spam Email Detector")
root.geometry("700x600")
root.resizable(False, False)
root.configure(bg="#F5F7FA")


# ==========================
# Logo
# ==========================
logo = tk.Label(
    root,
    text="📧",
    font=("Arial", 40),
    bg="#F5F7FA"
)
logo.pack(pady=(10, 0))


# ==========================
# Heading
# ==========================
heading = tk.Label(
    root,
    text="AI Powered Spam Email Detector",
    font=("Arial", 22, "bold"),
    bg="#F5F7FA",
    fg="#222222"
)
heading.pack(pady=5)


# ==========================
# Instruction
# ==========================
instruction = tk.Label(
    root,
    text="Enter your message below:",
    font=("Arial", 13),
    bg="#F5F7FA"
)
instruction.pack(pady=10)


# ==========================
# Text Box
# ==========================
text_box = tk.Text(
    root,
    height=8,
    width=65,
    font=("Arial", 11),
    relief="solid",
    bd=2
)
text_box.pack()


# ==========================
# Buttons
# ==========================
button_frame = tk.Frame(root, bg="#F5F7FA")
button_frame.pack(pady=20)


check_button = tk.Button(
    button_frame,
    text="Check Message",
    command=check_message,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12, "bold"),
    width=15,
    cursor="hand2"
)
check_button.grid(row=0, column=0, padx=10)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_text,
    bg="#F44336",
    fg="white",
    font=("Arial", 12, "bold"),
    width=10,
    cursor="hand2"
)
clear_button.grid(row=0, column=1, padx=10)


# ==========================
# Result
# ==========================
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 20, "bold"),
    bg="#F5F7FA"
)
result_label.pack(pady=20)


# ==========================
# Footer
# ==========================
footer = tk.Label(
    root,
    text="Developed by Kunal | Python • Scikit-learn • Naive Bayes • TF-IDF",
    font=("Arial", 10),
    fg="gray",
    bg="#F5F7FA"
)
footer.pack(side="bottom", pady=15)


# ==========================
# Run GUI
# ==========================
root.mainloop()