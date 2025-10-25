"""
Streamlit app for interacting with the LLMChatApp.    
"""
import streamlit as st
from main import LLMChatApp
from model_util import is_groq_model, SUPPORTED_GROQ_MODELS, SUPPORTED_OPENAI_MODELS

# Page Configuration
st.set_page_config(
    page_title="LLM Chat Application",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_app" not in st.session_state:
    st.session_state.llm_app = None
    
# Title and Description
st.title("Grok/OpenAI LLM Chat Application")
st.markdown("Chat with Groq or OpenAI language models directly from your browser!")
st.markdown("Please enter your API key in the sidebar to get started. You can get one at https://console.groq.com/ for Groq API keys or https://platform.openai.com/account/api-keys for OpenAI API keys.")

# Implement sidebar for configurations
with st.sidebar:
    st.header("Model Configuration")
    
    # Model Selection
    model = st.selectbox(
        "Select Model",
        options=SUPPORTED_GROQ_MODELS + SUPPORTED_OPENAI_MODELS,
        help="Choose a Groq or OpenAI model to use."
    )
    
    model_name = st.text_input(
        "Model Name",
        placeholder="Tidal",
        help="Provide a custom name for the model."
    )
    
    system_prompt = st.text_area(
        "System Prompt (Optional)",
        placeholder="You are a helpful assistant.",
        help="Provide a custom system prompt to guide the model's behavior."
    )
    
    api_key = st.text_input(
        f"{"Groq" if is_groq_model(model) else "OpenAI"} API Key",
        type="password",
        placeholder="*********",
        help=f"Enter your {"Groq" if is_groq_model(model) else "OpenAI"} API key here."
    )

    if st.button("Initialize LLM Chat App", use_container_width=True):
        if st.session_state.llm_app:
            st.session_state.llm_app.clear_history()
        st.session_state.messages = []
        try:
            st.session_state.llm_app = LLMChatApp(
                model=model,
                model_name=model_name,
                system_prompt=system_prompt,
                api_key=api_key
            )
            st.success("LLM Chat App initialized successfully!")
        except Exception as e:
            st.error(f"Error initializing LLM Chat App: {e}")
            
    st.header("Chat Configuration")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Select a value between 0-2 to control response randomness. Higher values make output more random"
    )
    
    max_tokens = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="Maximum number of tokens in the response."
    )
    
    if st.button("Clear Chat History", use_container_width=True):
        if st.session_state.llm_app:
            st.session_state.llm_app.clear_history()
            st.session_state.messages = []
            st.success("Chat history cleared!")
        else:
            st.error("Please initialize the LLM Chat App first.")
            
# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything..."):
    if not st.session_state.llm_app:
        st.error("Please initialize the LLM Chat App first.", icon="‼️")
    else:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"{prompt}"
            }
        )
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.llm_app.chat(
                        user_message=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    st.markdown(response)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"{response}"
                        }
                    )
                except Exception as e:
                    st.error(f"Error during response generation: {str(e)}")
