from langgraph.graph.state import CompiledStateGraph
import streamlit as st
from streamlit_chat import message

class DisplayResult:
    def __init__(self, user_controls, graph: CompiledStateGraph):
        self.usecase = user_controls["usecase"]
        self.graph = graph
        self.user_message = user_controls["user_message"]
        self.language = user_controls["language"] if self.usecase == "Blog Generation with Language Translation" else ""

    def display_result(self):
        if self.usecase == "Blog Generation":
            message(self.user_message, is_user=True)
            with st.spinner("Generating blog... ⏳"):
                response = self.graph.invoke({"topic": self.user_message})
                response_string = f"""
                Title:\n{response['blog'].title}\n
                Content:\n{response['blog'].content}
                """
                message(response_string, is_user=False)
        elif self.usecase == "Blog Generation with Language Translation":
            message(self.user_message, is_user=True)
            with st.spinner("Generating blog and translating... ⏳"):
                response = self.graph.invoke({"topic": self.user_message, "language": self.language})
                if response["language_check"] == "Valid language":
                    response_string = f"""
                    Original Blog:\n
                    Title:\n{response['blog'].title}\n
                    Content:\n {response['blog'].content}\n
                    Translated Blog:\n
                    Title:\n {response['blog'].translated_title}\n
                    Content:\n {response['blog'].translated_content}
                    """
                    message(response_string, is_user=False)
                else:
                    response_string = f"The provided language: {response['language']} is incorrect. Please enter a valid language."
                    message(response_string, is_user=False)
        else:
            st.error("Invalid use case selected")
