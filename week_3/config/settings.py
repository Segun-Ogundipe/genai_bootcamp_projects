import os

from dotenv import load_dotenv

load_dotenv(override=True)

class EnvConfig:
    """Configuration class to manage environment configurations."""

    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.huggingface_api_key = os.getenv("HF_API_KEY")

env_config = EnvConfig()
