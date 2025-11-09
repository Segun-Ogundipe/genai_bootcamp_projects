from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

import config.settings as env_config
from utils.model_util import SUPPORTED_EMBEDDING_PROVIDERS, SUPPORTED_OPENAI_EMBEDDING_MODELS, SUPPORTED_HUGGINGFACE_EMBEDDING_MODELS

class EmbeddingClient:
    def __init__(
        self,
        provider=SUPPORTED_EMBEDDING_PROVIDERS[0],
        model_name=SUPPORTED_OPENAI_EMBEDDING_MODELS[0],
        api_key: Optional[str]=None
    ):
        """
        Initialize the EmbeddingClient.
        
        Args:
            provider (str): The provider to use for the embedding model. Defaults to "openai" ("openai" or "huggingface").
            model_name (str): The name of the model to use for the embedding model. Defaults to "text-embedding-3-small".
            api_key (Optional[str]): Provider API key (falls back to environment variable if not provided).
        """
        self.provider = provider
        self.model_name = model_name
        self.__api_key = api_key or self.__get_api_key()
        self.embedder = self.__initialize_embedder()

    def __get_api_key(self) -> str:
        """Get API key from environment variables"""
        if self.model_name in SUPPORTED_OPENAI_EMBEDDING_MODELS:
            api_key = env_config.openai_api_key
            if not api_key:
                raise ValueError(f"API key is not set for the selected embedding model: {self.model_name}")
            return api_key

    def __initialize_embedder(self) -> Embeddings:
        """Create the embedding model instance based on the provider and model_name"""
        if self.provider == SUPPORTED_EMBEDDING_PROVIDERS[0]:
            return OpenAIEmbeddings(model=self.model_name, api_key=self.__api_key)
        elif self.provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
        
    def generate_embeddings(self, texts: List[Document]) -> List[float]:
        """Generate embeddings for a list of text documents"""
        return self.embedder.embed_documents(texts)
    