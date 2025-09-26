# ui_streamlit.py
import requests
import streamlit as st
import os 

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/advise")

st.set_page_config(page_title="UCD Advisor", page_icon="🎓", layout="centered")
st.title("🎓 UCD Advisor (Local Llama RAG)")

with st.form("advisor"):
    q = st.text_area("Ask about schedules, prerequisites, or course options:", height=140,
                     placeholder="e.g., Build me a 9–12 credit Fall schedule, mornings only, no conflicts.")
    submitted = st.form_submit_button("Ask")

if submitted and q.strip():
    with st.spinner("Thinking..."):
        r = requests.get(API_URL, params={"q": q})
    if r.status_code == 200:
        data = r.json()
        st.subheader("Advisor")
        st.write(data["answer"])

        st.subheader("Validation")
        st.write(f"**Suggested IDs:** {data['suggested_course_ids']}")
        st.write(f"**Total credits:** {data['total_credits']}")
        st.write(f"**Time conflict:** {data['time_conflict']}")
        if data["conflict_info"]:
            st.write(f"**Conflict details:** {data['conflict_info']}")
        if data["unmet_prerequisites"]:
            st.write("**Unmet prerequisites:**")
            st.write(data["unmet_prerequisites"])
    else:
        st.error(f"API error {r.status_code}: {r.text}")
