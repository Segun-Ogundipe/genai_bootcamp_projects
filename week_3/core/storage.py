import os
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    YoutubeLoader,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.embeddings import EmbeddingClient
from modules.news_summarizer.articleloader import ArticleLoader
from utils.model_util import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_OPENAI_EMBEDDING_MODELS
)

PERSIST_DIRECTORY = "data/chroma_db"

class VectorStore:
    def __init__(
        self,
        collection_name: str="collection",
        embedding_provider: str=SUPPORTED_EMBEDDING_PROVIDERS[0],
        embedding_model_name: str=SUPPORTED_OPENAI_EMBEDDING_MODELS[0],
        embedding_api_key: Optional[str]=None,
        chunk_size: int=1024,
        chunk_overlap: int=200
    ):
        """
        Initialize the VectorStore.

        Args:
            collection_name (str): The name of the collection to use for the vector store.
            embedding_provider (str): The provider to use for the embedding model.
            embedding_model_name (str): The name of the embedding model to use.
            embedding_api_key (Optional[str]): The API key to use for the embedding model.
            chunk_size (int): The size of the chunks to split the documents into.
            chunk_overlap (int): The overlap between the chunks.
        """
        self.collection_name = embedding_provider + "-" + collection_name
        self.embeddingClient = EmbeddingClient(
            provider=embedding_provider,
            model_name=embedding_model_name,
            api_key=embedding_api_key
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.store = None

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(PERSIST_DIRECTORY), exist_ok=True)

        # Try to load existing store
        if os.path.exists(PERSIST_DIRECTORY):
            self.load_store()

    def load_store(self) -> None:
        """Load vector store from the data directory"""
        self.store = Chroma(
            collection_name=self.collection_name,
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=self.embeddingClient.embedder
        )
    
    def load_document(
        self,
        source: str,
        source_type: str="pdf"
    ) -> List[Document]:
        """
        Load a document from a source

        Args:
            source (str): The source to load the document from.
            source_type (str): The type of the source ("pdf", "txt", "md", "yt", "article). Defaults to "pdf".

        Returns:
            List[Document]: List of documents loaded from the source.
        """
        if source_type == "pdf":
            loader = PyPDFLoader(source)
        elif source_type == "txt":
            loader = TextLoader(source)
        elif source_type == "md":
            loader = UnstructuredMarkdownLoader(source)
        elif source_type == "youtube":
            loader = YoutubeLoader.from_youtube_url(source)
        elif source_type == "news":
            loader = ArticleLoader(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

        return loader.load()

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process a list of documents by splitting them into chunks

        Args:
            documents (List[Document]): The documents to process.

        Returns:
            List[Document]: List of processed documents.
        """
        # Clean document metadata
        cleaned_documents = filter_complex_metadata(documents)

        # Split the documents into chunks
        chunks = self.text_splitter.split_documents(cleaned_documents)

        return chunks

    def create_store(self, documents: List[Document]) -> None:
        """
        Create a new vector store from a list of documents

        Args:
            documents (List[Document]): The documents to save to the store.
            metadata (Optional[Dict]): Additional metadata for the documents.
        """
        # Process the documents
        processed_documents = self.process_documents(documents)

        # Create the store
        self.store = Chroma.from_documents(
            documents=processed_documents,
            embedding=self.embeddingClient.embedder,
            collection_name=self.collection_name,
            persist_directory=PERSIST_DIRECTORY
        )

    def add_to_store(self, documents: List[Document]) -> None:
        """
        Add documents to the existing vector store

        Args:
            documents (List[Document]): The documents to add to the store.
            metadata (Optional[Dict]): Additional metadata for the documents.
        """
        if not self.store:
            raise RuntimeError("Vector store not found. Please create or load the store first.")

        # Process the documents
        processed_documents = self.process_documents(documents)

        # Add the documents to the store
        self.store.add_documents(processed_documents)

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        """
        Get the vector store as a retriever object

        Args:
            **kwargs: Additional keyword arguments to pass to the retriever.

        Returns:
            VectorStoreRetriever: A vector store retriever instance.
        """
        if not self.store:
            raise RuntimeError("Vector store not found. Please create or load the store first.")

        return self.store.as_retriever(**kwargs)
