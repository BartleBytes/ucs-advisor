# 🎓 UCD Advisor

An AI-powered course advisor built with **LangChain**, **Ollama**, and **ChromaDB**.  
It suggests schedules, checks for time conflicts, and validates prerequisites using local Llama models.

---

## 🚀 Demo

<video src="docs/demo.mp4" width="800" autoplay loop muted playsinline poster="docs/demo_poster.jpg">
  Your browser does not support the video tag.
</video>

*(Video auto-plays like a GIF. Open this README on GitHub to see it in action.)*

---

## ✨ Features

- 📚 Retrieves courses from a Chroma index
- 📅 Suggests schedules with course IDs and reasoning
- ⏰ Validates time conflicts and prerequisites
- 💻 Works offline with Ollama (`llama3`, `nomic-embed-text`)
- 🌐 Run as a **FastAPI backend**, **CLI app**, or optional **web UI**
- 🎥 Demo video included in `docs/`

---

## 🛠️ Requirements

- **Python** 3.9+
- **pip** or **conda**
- **Ollama** (installed from [ollama.ai](https://ollama.ai))
- (Optional) **Node.js** 18+ (for React UI)
- (Optional) **Streamlit** (for quick local UI)

---

## 📦 Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/YOUR-USERNAME/ucd-advisor.git
cd ucd-advisor
pip install -r requirements.txt
