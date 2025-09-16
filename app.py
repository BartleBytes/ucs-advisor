import os
import re
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI
import uvicorn

# ---------- env & data ----------
load_dotenv()

BACKEND = os.getenv("LLM_BACKEND", "ollama")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_OPENAI_API_KEY")

courses = pd.read_csv("data/courses.csv")

# ---------- helpers ----------
def to_minutes(hhmm: str) -> int:
    t = datetime.strptime(hhmm, "%H:%M")
    return t.hour * 60 + t.minute

def has_time_conflict(df: pd.DataFrame, chosen_ids: list[str]):
    chosen = df[df["course_id"].isin(chosen_ids)].copy()

    # Normalize times to minutes for easy overlap checks
    chosen["start_m"] = chosen["start_time"].apply(to_minutes)
    chosen["end_m"] = chosen["end_time"].apply(to_minutes)

    # Very simple day-bucket approach; expand as needed
    day_patterns = chosen["days"].unique()
    for dayset in day_patterns:
        subset = chosen[chosen["days"] == dayset].sort_values("start_m")
        last_end = None
        last_id = None
        for _, r in subset.iterrows():
            if last_end is not None and r["start_m"] < last_end:
                return True, (last_id, r["course_id"], dayset)
            last_end = r["end_m"]
            last_id = r["course_id"]
    return False, None

def unmet_prereqs(df: pd.DataFrame, chosen_ids: list[str], prior_taken: set[str] | None = None):
    if prior_taken is None:
        prior_taken = set()
    missing = []
    have_now = set(chosen_ids)
    for cid in chosen_ids:
        row = df[df["course_id"] == cid].iloc[0]
        prereq = str(row.get("prerequisites", "")).strip()
        if prereq:
            needed = [p.strip() for p in prereq.split(",") if p.strip()]
            for p in needed:
                if p not in have_now and p not in prior_taken:
                    missing.append((cid, p))
    return missing

def parse_course_ids(text: str):
    # e.g., CSCI-2312, MATH-1401
    return list(set(re.findall(r"[A-Z]{2,5}-\d{3,4}", text)))

# ---------- retriever ----------
emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
vs = Chroma(persist_directory="chroma_ucd", embedding_function=emb)
# Warn if the vector store has no documents
try:
    if len(vs.get().get("ids", [])) == 0:
        print("[warn] No embeddings found in 'chroma_ucd'. Run: python ingest.py")
except Exception:
    pass
retriever = vs.as_retriever(search_kwargs={"k": 6})

# ---------- model selection ----------
if BACKEND == "openai":
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.2)
else:
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.2, base_url=OLLAMA_HOST)

# ---------- prompt & chain ----------
SYSTEM = """You are a helpful academic advisor for University of Colorado Denver.
You must base answers only on the 'Context' provided; if information is missing, ask for the missing details or say you need more data.
When proposing schedules, list course_id, title, days, time, and credits.
Default to 9–12 credits unless the student asks otherwise.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Student request: {question}\n\nContext:\n{context}")
])

def rag_answer(question: str) -> str:
    docs = retriever.invoke(question)
    ctx = "\n\n".join(d.page_content for d in docs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question, "context": ctx})

# ---------- CLI ----------
def cli():
    print("UCD Advisor (local Llama). Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        ans = rag_answer(q)
        print("\nAdvisor:", ans)

        # Validate suggested schedule (if any course IDs detected)
        ids = [i for i in parse_course_ids(ans) if i in set(courses["course_id"])]
        if ids:
            conflict, info = has_time_conflict(courses, ids)
            missing = unmet_prereqs(courses, ids)
            total_credits = courses[courses["course_id"].isin(ids)]["credits"].sum()

            print("\n--- Validation ---")
            print("Proposed:", ids)
            print("Total credits:", float(total_credits))
            if conflict:
                a, b, dayset = info
                print(f"Time conflict between {a} and {b} on {dayset}")
            if missing:
                print("Unmet prerequisites:", missing)
            if (not conflict) and (not missing):
                print("Schedule appears feasible based on catalog data.")

# ---------- FastAPI (optional) ----------
api = FastAPI()

@api.get("/advise")
def advise(q: str):
    ans = rag_answer(q)
    ids = [i for i in parse_course_ids(ans) if i in set(courses["course_id"])]
    conflict, info = has_time_conflict(courses, ids)
    missing = unmet_prereqs(courses, ids)
    total = courses[courses["course_id"].isin(ids)]["credits"].sum()
    return {
        "answer": ans,
        "suggested_course_ids": ids,
        "total_credits": float(total),
        "time_conflict": conflict,
        "conflict_info": info,
        "unmet_prerequisites": missing,
    }

if __name__ == "__main__":
    # Choose one: CLI or API
    # uvicorn.run(api, host="0.0.0.0", port=8000)
    cli()
