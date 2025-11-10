import os

from dotenv import load_dotenv
import streamlit as st

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv(override=True)

welcome_page = st.Page("pages/welcome_page.py", title="AI Content Summarizer", icon="👋")
article_page = st.Page("pages/news_article_summarizer.py", title="Article Summarizer", icon="📰")
video_page = st.Page("pages/youtube_summarizer.py", title="YouTube Video Summarizer", icon="📺")
document_page = st.Page("pages/document_summarizer.py", title="Document Summarizer", icon="📄")

pg = st.navigation([welcome_page, article_page, video_page, document_page], position="top")

st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="🤖",
    layout="wide"
)

pg.run()
