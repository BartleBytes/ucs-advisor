import os
import re
from datetime import datetime
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from data_loaders import load_course_catalog

# ---------- env & data ----------
load_dotenv()

BACKEND = os.getenv("LLM_BACKEND", "openai").lower()  
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_OPENAI_API_KEY")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_ucd")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")


COURSE_CATALOG_PATH = os.getenv(
    "COURSE_CATALOG_PATH",
    "sources/Fall_2005_Class_List/Fall 2025 Class List_ZW.xlsx",
)
courses = load_course_catalog(COURSE_CATALOG_PATH)

# ---------- helpers ----------
def to_minutes(hhmm: str | None) -> Optional[int]:
    # Return minutes for valid "HH:MM" strings; tolerate blanks / malformed data.
    if not hhmm:
        return None
    try:
        t = datetime.strptime(hhmm.strip(), "%H:%M")
    except (ValueError, AttributeError):
        return None
    return t.hour * 60 + t.minute

def has_time_conflict(df: pd.DataFrame, chosen_ids: list[str]):
    chosen = df[df["course_id"].isin(chosen_ids)].copy()

    # Normalize times to minutes for easy overlap checks
    chosen["start_m"] = chosen["start_time"].apply(to_minutes)
    chosen["end_m"] = chosen["end_time"].apply(to_minutes)
    chosen = chosen[pd.notna(chosen["start_m"]) & pd.notna(chosen["end_m"])].copy()

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
        rows = df[df["course_id"] == cid]
        if rows.empty:
            continue  # skip unknown IDs
        row = rows.iloc[0]
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
if BACKEND == "openai":
    emb = OpenAIEmbeddings(model=EMBED_MODEL, api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=CHAT_MODEL, temperature=0.2)
else:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.chat_models import ChatOllama
    emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_HOST)
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.2, base_url=OLLAMA_HOST)

vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=emb)
retriever = vs.as_retriever(search_kwargs={"k": 6})

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
    print("UCD Advisor (OpenAI). Type 'exit' to quit.")
    id_set = set(courses["course_id"])
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        ans = rag_answer(q)
        print("\nAdvisor:", ans)

        # Validate suggested schedule (if any course IDs detected)
        ids = [i for i in parse_course_ids(ans) if i in id_set]
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

if ALLOWED_ORIGINS.strip() == "*":
    cors_origins = ["*"]
else:
    cors_origins = [origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()]

api.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api.get("/advise")
def advise(q: str):
    ans = rag_answer(q)
    id_set = set(courses["course_id"])
    ids = [i for i in parse_course_ids(ans) if i in id_set]
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
    # cli()
    uvicorn.run(api, host="0.0.0.0", port=8000)
