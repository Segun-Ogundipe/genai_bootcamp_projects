import os

import streamlit as st

from src.ui.config.ui_config_file import Config

class LoadUI:
    def __init__(self):
        self.config = Config()
        self.user_controls = {}

    def load_ui(self):
        st.set_page_config(page_title=self.config.get_page_title(), layout="wide", page_icon=":robot_face:")
        st.header("🤖 " + self.config.get_page_title())

        # Sidebar
        with st.sidebar:
            # Get options from config
            llm_options = self.config.get_llm_options()

            usecase_options = self.config.get_usecase_options()
            openai_model_options = self.config.get_openai_model_options()
            groq_model_options = self.config.get_groq_model_options()

            # Create dropdowns
            self.user_controls["llm"] = st.selectbox("Select LLM", llm_options)
            if self.user_controls["llm"] == "OpenAI":
                self.user_controls["openai_model"] = st.selectbox("Select OpenAI Model", openai_model_options)
                self.user_controls["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"] = st.text_input("Enter OpenAI API Key", type="password")
                if not self.user_controls["OPENAI_API_KEY"]:
                    st.warning("⚠️ OpenAI API Key is required")

            elif self.user_controls["llm"] == "Groq":
                self.user_controls["groq_model"] = st.selectbox("Select Groq Model", groq_model_options)
                self.user_controls["GROQ_API_KEY"] = st.session_state["GROQ_API_KEY"] = st.text_input("Enter Groq API Key", type="password")
                if not self.user_controls["GROQ_API_KEY"]:
                    st.warning("⚠️ Groq API Key is required")
            else:
                st.warning("⚠️ Invalid LLM selected")

            self.user_controls["usecase"] = st.selectbox("Select Use Case", usecase_options)
            if self.user_controls["usecase"] == "Blog Generation with Language Translation":
                self.user_controls["language"] = st.text_input("Enter Language", placeholder="French")
                if not self.user_controls["language"]:
                    st.warning("⚠️ Language is required")
        return self.user_controls
