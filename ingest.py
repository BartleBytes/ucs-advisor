import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def main():
    load_dotenv()
    embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

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
    emb = OllamaEmbeddings(model=embed_model, base_url=ollama_host)
    persist_dir = "chroma_ucd"
    vs = Chroma.from_documents(docs, emb, persist_directory=persist_dir)
    vs.persist()
    print(f"Index built: {persist_dir}/")

if __name__ == "__main__":
    main()
