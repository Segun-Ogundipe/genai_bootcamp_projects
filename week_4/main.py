import streamlit as st

from src.graphs.graph import GraphBuilder
from src.models.groq import GroqModel
from src.models.openai import OpenAIModel
from src.ui.display_result import DisplayResult
from src.ui.load_ui import LoadUI

def load_app():
    """
    Loads and runs the application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.

    """

    ##Load UI
    ui = LoadUI()
    user_controls = ui.load_ui()

    if not user_controls:
        st.error("Error: Failed to load user input from the UI.")
        return
    
    # Text input for user message
    user_message = st.chat_input("Enter the topic for the blog:")

    if user_message:
        try:
            ## Configure The LLM's
            llm_config = GroqModel(user_controls) if user_controls["llm"] == "Groq" else OpenAIModel(user_controls)
            model = llm_config.get_model()

            if not model:
                st.error("Error: LLM model could not be initialized")
                return
            
            # Initialize and set up the graph based on use case
            usecase = user_controls["usecase"]

            if not usecase:
                    st.error("Error: No use case selected.")
                    return
            
            ## Graph Builder
            graph_builder = GraphBuilder(model)
            try:
                graph = graph_builder.setup_graph(usecase)
                user_controls["user_message"] = user_message
                DisplayResult(user_controls, graph).display_result()
            except Exception as e:
                st.error(f"Error: Failed to display result: {e}")
                return
        except Exception as e:
            st.error(f"Error: Failed to load app: {e}")
            return

if __name__ == "__main__":
    load_app()