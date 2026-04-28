import streamlit as st
import joblib

# =============================
# 🎨 STYLING
# =============================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Playfair+Display:wght@600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 17px;
    background-color: #f8fafc;
}

/* Header full width */
.header {
    width: 100vw;
    margin-left: -50vw;
    left: 50%;
    position: relative;

    background: linear-gradient(135deg, #6366f1, #818cf8);
    padding: 50px 60px;
    color: white;
}

.header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 42px;
    margin-bottom: 10px;
}

.header p {
    font-size: 18px;
    opacity: 0.9;
}

/* Content container */
.block-container {
    max-width: 880px;
}

/* Cards */
.card {
    background: white;
    padding: 22px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 18px;
    transition: 0.2s;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.05);
}

/* Result card */
.result-card {
    background: #eef2ff;
    padding: 22px;
    border-radius: 12px;
    border-left: 6px solid #6366f1;
}

/* Section headings */
h2, h3 {
    color: #1e293b;
}

/* Feature highlights */
.feature {
    padding: 10px;
    border-radius: 8px;
    background: #f1f5ff;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border-radius: 8px;
    padding: 10px 18px;
    border: none;
    font-weight: 500;
}

.stButton button:hover {
    background: linear-gradient(135deg, #4f46e5, #4338ca);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #f1f5f9;
}
</style>
""", unsafe_allow_html=True)

# =============================
# 🧠 LOAD MODEL
# =============================
model = joblib.load("model/model.pkl")

# =============================
# 🔁 PAGE STATE (FIXED PROPERLY)
# =============================
if "page" not in st.session_state:
    st.session_state.page = "Home"

selected_page = st.sidebar.radio(
    "Navigate",
    ["Home", "Symptom Checker"],
    index=["Home", "Symptom Checker"].index(st.session_state.page)
)

# Only update when user changes sidebar
if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

page = st.session_state.page

# =============================
# 🏠 HOME PAGE
# =============================
if page == "Home":

    st.markdown("""
    <div class="header">
        <h1>Health Insight</h1>
        <p>Describe how you're feeling and get a quick assessment.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### Why use this?")
    st.write("Get quick, AI-based insights from your symptoms using natural language.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Natural Input**")
        st.write("Type symptoms freely")

    with col2:
        st.markdown("**Instant Results**")
        st.write("Get quick predictions")

    with col3:
        st.markdown("**Explainable Output**")
        st.write("Understand the result")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### How it works")
    st.write("""
    1. Describe your symptoms  
    2. Model analyzes patterns  
    3. Get condition + suggestions  
    """)

    if st.button("Start Symptom Check"):
        st.session_state.page = "Symptom Checker"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 🧠 SYMPTOM CHECKER
# =============================
elif page == "Symptom Checker":

    st.markdown("""
    <div class="header">
        <h1>Symptom Checker</h1>
        <p>Enter your symptoms below</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    user_input = st.text_area(
        "Example: I have fever and cough for 3 days",
        height=120
    )

    analyze_clicked = st.button("Analyze")

    st.markdown('</div>', unsafe_allow_html=True)

    if analyze_clicked:

        if not user_input.strip():
            st.error("Please enter your symptoms.")
            st.stop()

        # 🔥 NLP Prediction
        prediction = model.predict([user_input])
        probabilities = model.predict_proba([user_input])

        disease = prediction[0]
        confidence = max(probabilities[0]) * 100

        classes = model.classes_
        top2_idx = probabilities[0].argsort()[-2:][::-1]

        # Severity
        text = user_input.lower()
        severity_score = sum(word in text for word in ["high", "severe", "very", "extreme"])

        if severity_score == 0:
            severity = "Mild"
        elif severity_score == 1:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Guidance
        guidance = {
            "Flu": {"advice": "Rest, fluids, avoid cold exposure", "meds": "Paracetamol, Cough syrup"},
            "Cold": {"advice": "Warm fluids, rest", "meds": "Decongestants"},
            "Viral Fever": {"advice": "Hydration + rest", "meds": "Paracetamol"},
            "Headache": {"advice": "Rest, reduce screen time", "meds": "Pain relievers"},
            "Allergy": {"advice": "Avoid allergens", "meds": "Antihistamines"}
        }

        # RESULT
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        st.markdown(f"## {disease}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.progress(int(confidence))

        st.markdown('</div>', unsafe_allow_html=True)

        # DETAILS
        st.markdown('<div class="card">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Severity**")
            st.write(severity)

        with col2:
            st.markdown("**Other possibilities**")
            for idx in top2_idx:
                st.write(f"{classes[idx]} ({probabilities[0][idx]*100:.2f}%)")

        st.markdown('</div>', unsafe_allow_html=True)

        # ADVICE
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("### What you can do")
        st.write(guidance.get(disease, {}).get("advice", "General care recommended"))

        st.markdown("### Typical treatments")
        st.write(guidance.get(disease, {}).get("meds", "Consult a doctor"))

        st.markdown('</div>', unsafe_allow_html=True)

        # EXPLANATION
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown("### Why this result")

        try:
            vectorizer = model.named_steps["vectorizer"]
            input_vector = vectorizer.transform([user_input]).toarray()[0]
            feature_names = vectorizer.get_feature_names_out()

            important_words = [
                feature_names[i] for i, val in enumerate(input_vector) if val > 0
            ]

            if important_words:
                st.write(
                    f"This result is based on symptoms like {', '.join(important_words[:5])}."
                )
            else:
                st.write("Prediction based on general patterns.")

        except:
            st.write("Explanation unavailable.")

        st.markdown('</div>', unsafe_allow_html=True)

        st.caption("This tool is for educational purposes only.")