import streamlit as st
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CyberShield AI Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("spam_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ---------------- HIDE STREAMLIT DEFAULT UI ----------------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

body {
    background: radial-gradient(circle at top, #0f2027, #000000);
    color: white;
}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 50px;
    margin-top: 40px;
    box-shadow: 0 0 80px rgba(0,255,255,0.12);
}

.title {
    text-align: center;
    font-size: 3.6rem;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    opacity: 0.85;
    margin-bottom: 40px;
    font-size: 1.1rem;
}

.result {
    margin-top: 35px;
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
}

.ham {
    background: linear-gradient(135deg, #00ff99, #00ccff);
    color: black;
}

.spam {
    background: linear-gradient(135deg, #ff3c3c, #ff0055);
}

.uncertain {
    background: linear-gradient(135deg, #ffcc00, #ff9900);
    color: black;
}

.prob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-top: 25px;
}

.prob-card {
    background: rgba(255,255,255,0.12);
    border-radius: 18px;
    padding: 25px;
    text-align: center;
    font-size: 1.3rem;
}

.footer {
    margin-top: 60px;
    text-align: center;
    opacity: 0.7;
    font-size: 0.95rem;
}
</style>

<div class="glass">
    <div class="title">🛡️ CyberShield</div>
    <div class="subtitle">
        AI-Powered Spam & Ham Classification with Confidence-Based Decisioning
    </div>
""", unsafe_allow_html=True)

# ---------------- INPUT ----------------
message = st.text_area(
    "Paste the email or message below",
    height=180,
    placeholder="Enter message text here..."
)

analyze = st.button("🔍 Detect Message Type")

# ---------------- PREDICTION ----------------
if analyze and message.strip():
    features = vectorizer.transform([message])
    probs = model.predict_proba(features)[0]

    spam_prob = probs[0] * 100
    ham_prob = probs[1] * 100

    if ham_prob >= 70:
        label = "HAM MESSAGE"
        css = "ham"
        icon = "✅"
    elif spam_prob >= 70:
        label = "SPAM MESSAGE"
        css = "spam"
        icon = "🚨"
    else:
        label = "UNCERTAIN MESSAGE"
        css = "uncertain"
        icon = "⚠️"

    st.markdown(f"""
    <div class="result {css}">
        {icon} {label}
    </div>

    <div class="prob-grid">
        <div class="prob-card">
            📩 Ham Probability<br><b>{ham_prob:.2f}%</b>
        </div>
        <div class="prob-card">
            🚫 Spam Probability<br><b>{spam_prob:.2f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif analyze:
    st.warning("Please enter a message for analysis.")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    Designed by <b>Gobika S</b>
</div>
</div>
""", unsafe_allow_html=True)

