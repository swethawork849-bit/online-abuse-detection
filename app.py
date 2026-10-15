
import streamlit as st
import pickle
import re
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ======================================================
# PAGE CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Online Abuse Detection System",
    page_icon="🛡️",
    layout="centered"
)

# ======================================================
# DOWNLOAD NLTK DATA
# ======================================================

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ======================================================
# LOAD MODEL & TOKENIZER
# ======================================================

model = load_model("models/online_abuse_bilstm.keras")

with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ======================================================
# NLP SETUP
# ======================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ======================================================
# APP HEADER
# ======================================================

st.title("🛡️ Online Abuse Detection System")
st.subheader("Deep Learning Mini Project using Bidirectional LSTM (BiLSTM)")

st.write("""
This application detects abusive comments using **Natural Language Processing (NLP)**
and a **Bidirectional LSTM (BiLSTM)** model.

### Unique Features
- 🚦 Abuse Severity Meter
- ☠️ Toxic Word Detection
- 🤝 Respectful Comment Suggestion
- 👤 Fake Account Identifier
""")

st.divider()

# ======================================================
# TEXT PREPROCESSING
# ======================================================

def clean_text(text):
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    # Remove stopwords
    words = [w for w in words if w not in stop_words]

    # Lemmatization
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

# ======================================================
# MODEL PREDICTION
# ======================================================

def predict_comment(comment):
    cleaned = clean_text(comment)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=200,
        padding="post",
        truncating="post"
    )

    probability = float(model.predict(padded, verbose=0)[0][0])

    return cleaned, probability

# ======================================================
# TOXIC WORD LIST
# ======================================================

TOXIC_WORDS = [
    "idiot",
    "stupid",
    "moron",
    "loser",
    "bitch",
    "fuck",
    "hate",
    "ugly",
    "useless",
    "dumb"
]

# ======================================================
# TOXIC WORD DETECTION
# ======================================================

def find_toxic_words(comment):
    words = re.findall(r"\b[a-zA-Z]+\b", comment.lower())

    found = []

    for word in words:
        if word in TOXIC_WORDS:
            found.append(word)

    return found

# ======================================================
# SEVERITY CLASSIFICATION
# ======================================================

def get_severity(probability, toxic_count):

    if probability < 0.50:
        return "🟢 Safe"

    elif toxic_count == 1:
        return "🟡 Mild Abuse"

    elif toxic_count == 2:
        return "🟠 Moderate Abuse"

    else:
        return "🔴 Severe Abuse"

# ======================================================
# RESPECTFUL COMMENT SUGGESTION
# ======================================================

def suggest_comment(comment):

    replacements = {
        "idiot": "person",
        "stupid": "unwise",
        "moron": "person",
        "loser": "friend",
        "bitch": "person",
        "fuck": "please",
        "hate": "dislike",
        "ugly": "different",
        "useless": "helpful",
        "dumb": "smart"
    }

    suggestion = comment

    for bad, good in replacements.items():
        suggestion = re.sub(
            rf"\b{bad}\b",
            good,
            suggestion,
            flags=re.IGNORECASE
        )

    return suggestion

# ======================================================
# FAKE ACCOUNT IDENTIFIER
# ======================================================

def fake_account_risk(comment):

    toxic_count = len(find_toxic_words(comment))

    if toxic_count == 0:
        return "🟢 Low Risk Account"

    elif toxic_count <= 2:
        return "🟡 Medium Risk Account"

    else:
        return "🔴 High Risk Account"

# ======================================================
# USER INPUT
# ======================================================

st.subheader("💬 Enter a Social Media Comment")

comment = st.text_area(
    "Type your comment below:",
    placeholder="Example: You are an idiot!",
    height=170
)

# ======================================================
# DETECT BUTTON
# ======================================================

if st.button("🚀 Detect Abuse", use_container_width=True):

    if comment.strip() == "":
        st.warning("Please enter a social media comment.")

    else:

        cleaned, probability = predict_comment(comment)

        toxic = find_toxic_words(comment)

        severity = get_severity(probability, len(toxic))

        prediction = (
            "🚨 Abusive Comment Detected"
            if probability >= 0.5
            else "✅ Non-Abusive Comment"
        )

        # ==================================================
        # PREDICTION RESULT
        # ==================================================

        st.header("🤖 AI Prediction Result")

        if probability >= 0.5:
            st.error(prediction)
        else:
            st.success(prediction)

        # ==================================================
        # PROBABILITY
        # ==================================================

        st.subheader("📊 Abuse Probability")

        st.progress(probability)

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Prediction",
                "Abusive" if probability >= 0.5 else "Non-Abusive"
            )

        with col2:
            st.metric(
                "Confidence Score",
                f"{probability*100:.2f}%"
            )

        # ==================================================
        # SEVERITY METER
        # ==================================================

        st.subheader("🚦 Abuse Severity Meter")

        if severity == "🟢 Safe":
            st.success(severity)

        elif severity == "🟡 Mild Abuse":
            st.warning(severity)

        elif severity == "🟠 Moderate Abuse":
            st.warning(severity)

        else:
            st.error(severity)

        # ==================================================
        # TOXIC WORDS
        # ==================================================

        st.subheader("☠️ Toxic Word Detection")

        if toxic:
            st.write("**Detected Toxic Words:**")

            cols = st.columns(min(len(toxic), 4))

            for i, word in enumerate(toxic):
                cols[i % 4].error(word.upper())

        else:
            st.success("No toxic words detected.")

        # ==================================================
        # CLEANED COMMENT
        # ==================================================

        st.subheader("🧹 Cleaned Comment Used by AI")

        st.code(cleaned)

        # ==================================================
        # RESPECTFUL COMMENT
        # ==================================================

        st.subheader("🤝 Respectful Comment Suggestion")

        st.success(suggest_comment(comment))

        # ==================================================
        # FAKE ACCOUNT IDENTIFIER
        # ==================================================

        st.subheader("👤 Fake Account Identifier")

        risk = fake_account_risk(comment)

        if "Low" in risk:
            st.success(risk)

        elif "Medium" in risk:
            st.warning(risk)

        else:
            st.error(risk)

        # ==================================================
        # AI SUMMARY
        # ==================================================

        st.divider()

        st.subheader("📋 AI Analysis Summary")

        summary = {
            "Feature": [
                "Prediction",
                "Abuse Probability",
                "Severity Level",
                "Toxic Words Found",
                "Respectful Suggestion",
                "Fake Account Risk"
            ],
            "Result": [
                "Abusive" if probability >= 0.5 else "Non-Abusive",
                f"{probability*100:.2f}%",
                severity,
                ", ".join(toxic) if toxic else "None",
                suggest_comment(comment),
                risk
            ]
        }

        st.table(summary)

# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption("💜 Developed by Swetha | AI & DS Mini Project")