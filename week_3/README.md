
# Week 3 — Generative AI / RAG Summarizers

This repository contains the Week 3 project from the "AI & LLM Engineering Mastery — GenAI + RAG" course. It provides a set of small summarization utilities and a lightweight app that demonstrates retrieval-augmented generation (RAG) workflows: ingesting documents or transcripts, building embeddings, and producing concise summaries via an LLM.

Key features
- Multiple summarizer: news articles, PDF documents, and YouTube transcripts.
- Modular core components for embeddings, LLM interaction, and storage.

Project layout

- `app.py` — lightweight UI / demo application (entrypoint)
- `config/settings.py` — configuration and environment-handling (API keys, provider settings)
- `core/` — core building blocks
	- `embeddings.py` — embeddings abstraction
	- `llm.py` — LLM / prompt wrapper
	- `storage.py` — local vector store interface
- `pages/` — UI for individual summarizers
	- `news_article_summarizer.py` — summarizer for news articles
	- `document_summarizer.py` — PDF document summarizer
	- `youtube_summarizer.py` — summarizer for YouTube videos
	- `welcome_page.py` — simple web entrypoint
- `summarizer/` — submodules for different data-types (news, pdf, youtube)
- `utils/` — utility helpers (models, voice, etc.)

Quick start (macOS / zsh)

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Provide API keys and configuration

Edit `config/settings.py` or create and set the appropriate environment variables for your LLM / embeddings provider in a `.env` file. Check `.env.example` for the apropriate keys.

4. Run the demo app or a single summarizer

Run the main demo (if implemented):

```bash
streamlit run app.py
```

Notes on configuration and keys
- The project keeps settings in `config/settings.py`. If you prefer environment variables, update that file to read from `os.environ`.
- If you use an external vector database (Pinecone, Weaviate, etc.), follow that provider's setup and ensure `core/storage.py` is configured to use it.
