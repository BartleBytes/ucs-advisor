# 🎓 UCD Advisor

An AI-powered course advisor built with **LangChain**, **Ollama**, and **ChromaDB**.  
Suggests schedules, checks time conflicts, and validates prerequisites using local Llama models.

---

## 🚀 Demo


![Demo](docs/demo.gif)


## ✨ Features
- 📚 RAG over a local course catalog (ChromaDB)
- 🤖 Local LLM via Ollama (`llama3`), embeddings via `nomic-embed-text`
- 🧮 Validates credits, detects time conflicts, checks prerequisites
- 🧰 Run as a CLI or FastAPI backend; optional Streamlit/React frontend

---

## 🛠️ Setup

```bash
pip install -r requirements.txt
ollama pull llama3
ollama pull nomic-embed-text
 
