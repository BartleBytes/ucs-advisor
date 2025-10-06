# ui_streamlit.py
import requests
import streamlit as st
import os 

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/advise")

SAMPLE_QUESTIONS = [
    "Select a sample question...",
    "What are the core courses in Marketing?",
    "What are the graduation requirements for Accounting?",
    "What is the course information for BANA 6620?",
    "What are the time and location details for ACCT 2220 in Fall 2025?",
    "What courses does Ziyi Wang teach in Fall 2025, and what is his email address?",
    "What is the prerequisite for ACCT 2220 ?",
    "What courses should a freshman in Business take in Semester 1?",
]

if "question_input" not in st.session_state:
    st.session_state["question_input"] = ""


def _set_question_from_sample():
    choice = st.session_state.get("sample_question", "")
    if choice and choice != SAMPLE_QUESTIONS[0]:
        st.session_state["question_input"] = choice

st.set_page_config(page_title="UCD Advisor", page_icon="🎓", layout="centered")
st.title("🎓 UCD Advisor (Streamlit RAG Application)")

st.selectbox(
    "Try one of these examples:",
    SAMPLE_QUESTIONS,
    key="sample_question",
    on_change=_set_question_from_sample,
)

with st.form("advisor"):
    q = st.text_area(
        "Ask about schedules, prerequisites, or course options:",
        height=140,
        placeholder="e.g., Build me a 9–12 credit Fall schedule, mornings only, no conflicts.",
        key="question_input",
    )
    submitted = st.form_submit_button("Ask")

question = st.session_state["question_input"].strip()

if submitted and question:
    with st.spinner("Thinking..."):
        r = requests.get(API_URL, params={"q": question})
    if r.status_code == 200:
        data = r.json()
        st.subheader("Advisor")
        st.write(data["answer"])

        # st.subheader("Validation")
        # st.write(f"**Suggested IDs:** {data['suggested_course_ids']}")
        # st.write(f"**Total credits:** {data['total_credits']}")
        # st.write(f"**Time conflict:** {data['time_conflict']}")
        # if data["conflict_info"]:
        #     st.write(f"**Conflict details:** {data['conflict_info']}")
        # if data["unmet_prerequisites"]:
        #     st.write("**Unmet prerequisites:**")
        #     st.write(data["unmet_prerequisites"])
    else:
        st.error(f"API error {r.status_code}: {r.text}")
