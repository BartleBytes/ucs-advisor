import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

def main():
    load_dotenv()
    embed_model = os.getenv("EMBED_MODEL", os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_OPENAI_API_KEY")


    # Load course data
    df = pd.read_csv("data/courses.csv")

    # Turn each row into a retrievable Document
    docs = []
    for _, r in df.iterrows():
        text = (
            f"course_id: {r['course_id']}\n"
            f"title: {r['title']}\n"
            f"credits: {r['credits']}\n"
            f"days: {r['days']}\n"
            f"time: {r['start_time']}-{r['end_time']}\n"
            f"instructor: {r['instructor']}\n"
            f"prerequisites: {r['prerequisites']}\n"
            f"term: {r['term']}\n"
            f"campus: {r['campus']}\n"
            f"modality: {r['modality']}\n"
        )
        docs.append(Document(page_content=text, metadata={"course_id": r["course_id"]}))

    # Build embeddings and persist to Chroma
    if backend == "openai":
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_BACKEND=openai")
        emb = OpenAIEmbeddings(model=embed_model, api_key=openai_key)
    else:
        emb = OllamaEmbeddings(model=embed_model, base_url=ollama_host)
    persist_dir = os.getenv("CHROMA_DIR", "chroma_ucd")

    vs = Chroma.from_documents(docs, emb, persist_directory=persist_dir)
    vs.persist()
    print(f"Index built: {persist_dir}/")

if __name__ == "__main__":
    main()
