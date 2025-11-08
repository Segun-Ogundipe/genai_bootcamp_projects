from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

def get_embedding_model(provider: str="openai", model_name: str="text-embedding-3-small", api_key: Optional[str]=None):
    """
    Returns an embedding model instance based on the specified provider and model name.

    Args:
        provider (str): The provider to use for the embedding model. Defaults to "openai" ("openai" or "huggingface").
        model_name (str): The name of the model to use for the embedding model. Defaults to "text-embedding-3-small".

    Returns:
        An embedding model instance.
    """
    # Raise an error if the API key is not provided
    # if api_key is None:
    #     raise ValueError("API key is required for the embedding model. Provide it via the `api_key` parameter or set the `OPENAI_API_KEY` or `GROQ_API_KEY` environment variable.")
    
    # Initialize the appropriate embedding model
    if provider == "openai":
        return OpenAIEmbeddings(model=model_name)
    elif provider == "huggingface":
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

def generate_embeddings(texts: List[Document], provider="openai", model_name="text-embedding-3-small", api_key: str=None) -> List[float]:
    """
    Generates embeddings for a list of text documents.

    Args:
        texts (List[Document]): List of text documents to generate embeddings for.
        provider (str): The provider to use for the embedding model. Defaults to "openai" ("openai" or "groq").
        model_name (str): The name of the model to use for the embedding model. Defaults to "text-embedding-3-small".
        api_key (str): The API key to use for the embedding model.

    Returns:
        List[float]: List of embedding vectors for the text documents.
    """
    embedder = get_embedding_model(provider, model_name, api_key)

    return embedder.embed_documents(texts)
    