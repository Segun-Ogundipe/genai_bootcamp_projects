import streamlit as st
from streamlit_chat import message

from summarizer.pdf_summarizer.summarizer import PDFSummarizer
from utils.model_util import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_GROQ_MODELS,
    SUPPORTED_OPENAI_MODELS,
    SUPPORTED_OPENAI_EMBEDDING_MODELS,
    SUPPORTED_HUGGINGFACE_EMBEDDING_MODELS
)
from utils.voice_util import VoiceProcessor

voice_processor = VoiceProcessor()

# Page Configuration
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="auto"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "summarizer" not in st.session_state:
    st.session_state.summarizer = None
    
# Title and Description
st.title("Document Summarizer �")
st.markdown("Please configure the llm and embedding model in the sidebar. You can get an API key at https://console.groq.com/ for Groq API keys or https://platform.openai.com/account/api-keys for OpenAI API keys.")
with st.expander("Summarize Document", expanded=True):
    st.markdown("Please upload a PDF document to get started.")
    uploaded_file = st.file_uploader("Upload Document", type=["pdf"])
    summary_type = st.selectbox("Summary Type", options=["Detailed", "Concise"])
    if st.button("Summarize Document"):
        with st.spinner("Summarizing document..."):
            try:
                summary = st.session_state.summarizer.summarize_document(uploaded_file, summary_type=summary_type)
                st.write(summary)
            except Exception as e:
                st.error(f"Error summarizing document: {e}")

# Implement sidebar for configurations
with st.sidebar:
    st.header("Document Summarizer Model Configuration")
    
    # Model Selection
    llm_provider = st.selectbox(
        "Select LLM Model (Optional)",
        options=SUPPORTED_LLM_PROVIDERS,
        help="Choose a LLM provider to use."
    )

    llm_name = st.selectbox(
        "Select LLM Model (Optional)",
        options=SUPPORTED_OPENAI_MODELS if llm_provider == "OpenAI" else SUPPORTED_GROQ_MODELS,
        help="Choose a LLM model to use."
    )
    
    api_key = st.text_input(
        f"{llm_provider} API Key",
        type="password",
        placeholder="*********",
        help=f"Enter your {llm_provider} API key here."
    )

    embedding_provider = st.selectbox(
        "Select Embedding Model (Optional)",
        options=SUPPORTED_EMBEDDING_PROVIDERS,
        help="Choose a embedding provider to use."
    )

    embedding_model_name = st.selectbox(
        "Select Embedding Model (Optional)",
        options=SUPPORTED_OPENAI_EMBEDDING_MODELS if embedding_provider == "OpenAI" else SUPPORTED_HUGGINGFACE_EMBEDDING_MODELS,
        help="Choose a embedding model to use."
    )

    if embedding_provider == SUPPORTED_EMBEDDING_PROVIDERS[0]:
        embedding_api_key = st.text_input(
            f"{embedding_provider} API Key",
            type="password",
            placeholder="*********",
            help=f"Enter your {embedding_provider} API key here.",
            key="embedding_api_key"
        )

    chunk_size = st.slider(
        "Chunk Size",
        min_value=256,
        max_value=2048,
        value=1024,
        step=256,
        help="The size of the chunks to split the documents into."
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=400,
        value=200,
        step=100,
        help="The overlap between the chunks."
    )

    if st.button("Initialize Document Summarizer", use_container_width=True):
        st.session_state.messages = []
        try:
            st.session_state.summarizer = PDFSummarizer(
                llm_provider=llm_provider,
                llm_name=llm_name,
                llm_api_key=api_key,
                embedding_provider=embedding_provider,
                embedding_model_name=embedding_model_name,
                embedding_api_key=embedding_api_key,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            st.success("Document Summarizer initialized successfully!")
        except Exception as e:
            print(e)
            st.error(f"Error initializing Document Summarizer: {e}")
    
    if st.button("Clear Document Summarizer", use_container_width=True):
        if st.session_state.summarizer:
            st.session_state.messages = []
            st.success("Document Summarizer cleared!")
        else:
            st.error("Please initialize the Document Summarizer first.")
            
# Display chat messages from history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        message(msg["content"], is_user=True)
    else:
        message(msg["content"])

# Chat input
if prompt := st.chat_input("Ask anything about the document..."):
    if not st.session_state.summarizer:
        st.error("Please initialize the Document Summarizer first.", icon="‼️")
    else:
        # Display user message
        message(prompt, is_user=True)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": f"{prompt}"
            }
        )
        
        # Get assistant response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.summarizer.generate_response(prompt)
                message(response)
                audio_bytes = voice_processor.text_to_speech(response)
                st.audio(audio_bytes)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"{response}"
                    }
                )
            except Exception as e:
                st.error(f"Error during response generation: {str(e)}")

# Audio Input
audio_value = st.audio_input("Ask anything about the document...", sample_rate=48000)
if audio_value:
    audio_text = voice_processor.transcribe(audio_value)
    message(audio_text, is_user=True, key="audio_text")
    st.session_state.messages.append(
        {
            "role": "user",
            "content": f"{audio_text}"
        }
    )
    with st.spinner("Thinking..."):
        try:
            response = st.session_state.summarizer.generate_response(audio_text)
            message(response, key="audio_response")
            audio_bytes = voice_processor.text_to_speech(response)
            st.audio(audio_bytes)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"{response}"
                }
            )
        except Exception as e:
            st.error(f"Error during response generation: {str(e)}")
                