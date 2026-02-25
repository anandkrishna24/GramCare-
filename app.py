import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="Pediatric Triage MVP", page_icon="🏥", layout="wide")

# Language Selection
st.sidebar.title("Settings")
lang_choice = st.sidebar.radio("Select Language / ഭാഷ തിരഞ്ഞെടുക്കുക:", ["English", "Malayalam"])
l_key = "en" if lang_choice == "English" else "ml"

st.title("🏥 Pediatric Triage System (MVP)")
st.caption("A decoupled architecture with FastAPI and Streamlit.")

# Fetch questions from backend
@st.cache_data
def get_questions():
    try:
        response = requests.get(f"{BACKEND_URL}/questions")
        return response.json()
    except:
        st.error("Cannot connect to backend server. Please ensures the FastAPI server is running.")
        return {}

QUESTIONS = get_questions()

if QUESTIONS:
    col1, col2 = st.columns([2, 1])
    answers = {}

    with col1:
        # Build UI dynamically from API questions
        for cat, qs in QUESTIONS.items():
            with st.expander(f"{cat}", expanded=True):
                for q_id, q_data in qs.items():
                    label = q_data.get(l_key, q_data.get("en", q_id))
                    
                    if q_data["type"] == "number":
                        answers[q_id] = st.number_input(label, min_value=q_data["min"], max_value=q_data["max"], key=q_id)
                    elif q_data["type"] == "radio":
                        options = q_data.get("options", {"Yes": {"en": "Yes", "ml": "അതെ"}, "No": {"en": "No", "ml": "അല്ല"}})
                        answers[q_id] = st.radio(
                            label,
                            options=list(options.keys()),
                            format_func=lambda x: options[x].get(l_key, x),
                            key=q_id, horizontal=True
                        )
                        if q_data.get("is_critical") and answers[q_id] == "Yes":
                            st.warning("⚠️ High risk flag.")

    with col2:
        st.subheader("Assessment / വിലയിരുത്തൽ")
        if st.button("Get Triage Result / ഫലം ലഭിക്കുക", use_container_width=True):
            with st.spinner("Analyzing via Backend..."):
                try:
                    payload = {"answers": answers, "language": l_key}
                    res = requests.post(f"{BACKEND_URL}/triage", json=payload)
                    data = res.json()
                    
                    # Display Results
                    level = data["triage_level"]
                    color = "red" if level == "RED" else "orange" if level == "YELLOW" else "green"
                    
                    st.markdown(f"### Triage Level: :{color}[{level}]")
                    st.info(data["reasoning"])
                    
                    if data.get("advice_texts"):
                        st.subheader("Home Care Advice")
                        for advice in data["advice_texts"]:
                            st.write(f"- {advice}")
                    
                    st.divider()
                    st.write(f"**Confidence:** {data['confidence']}")
                except Exception as e:
                    st.error(f"Error fetching triage result: {e}")

st.divider()
st.info("Disclaimer: This tool is for informational purposes only.")
