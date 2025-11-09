from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from core.llm import LLMClient
from core.storage import VectorStore
from utils.model_util import (
    SUPPORTED_EMBEDDING_PROVIDERS,
    SUPPORTED_LLM_PROVIDERS,
    SUPPORTED_OPENAI_MODELS,
    SUPPORTED_OPENAI_EMBEDDING_MODELS
)

class YoutubeSummarizer:
    def __init__(
        self,
        llm_provider: str=SUPPORTED_LLM_PROVIDERS[0],
        llm_name: str=SUPPORTED_OPENAI_MODELS[0],
        llm_api_key: Optional[str]=None,
        embedding_provider: str=SUPPORTED_EMBEDDING_PROVIDERS[0],
        embedding_model_name: str=SUPPORTED_OPENAI_EMBEDDING_MODELS[0],
        embedding_api_key: Optional[str]=None,
        chunk_size: int=1024,
        chunk_overlap: int=200,
    ):
        """
        Initialize the YoutubeSummarizer with choice of model.

        Args:
            llm_provider (str): The provider to use for the LLM (openai or groq).
            llm_name (str): The name of the LLM to use.
            llm_api_key (Optional[str]): The API key to use for the LLM.
            embedding_provider (str): The provider to use for the embedding model (openai or huggingface).
            embedding_model_name (str): The name of the embedding model to use.
            embedding_api_key (Optional[str]): The API key to use for the embedding model.
        """
        self.llm_provider = llm_provider
        self.llm_name = llm_name
        self.llm_api_key = llm_api_key

        self.embedding_provider = embedding_provider
        self.embedding_model_name = embedding_model_name
        self.embedding_api_key = embedding_api_key

        self.store = VectorStore(
            collection_name="youtube-store",
            embedding_provider=self.embedding_provider,
            embedding_model_name=self.embedding_model_name,
            embedding_api_key=self.embedding_api_key,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.client = LLMClient(
            provider=self.llm_provider,
            model_name=self.llm_name,
            api_key=self.llm_api_key,
            store=self.store,
        )

    def download_and_process_video(self, url: str) -> List[Document]:
        """
        Download and process a YouTube video from a URL.

        Args:
            url (str): The URL of the YouTube video to download and process.
        """
        documents = self.store.load_document(source=url, source_type="youtube")

        try:
            processed_documents = self.store.add_to_store(documents)
        except Exception:
            processed_documents = self.store.create_store(documents)

        return processed_documents

    def summarize_video(self, url: str, summary_type: str="concise") -> str:
        """
        Summarize a YouTube video from a URL.

        Args:
            url (str): The URL of the YouTube video to summarize.
            summary_type (str): The type of summary to generate ("detailed" or "concise"). Defaults to "concise".
        """
        documents = self.download_and_process_video(url)

        if summary_type == "Detailed":
            map_prompt_template = """
            Write a detailed summary of the following video transcript segments:
            
            "{segments}"
            """
            
            combine_prompt_template = """
            Write a detailed summary of the following video transcript that combines the previous summaries:
            
            "{transcript}"
            """
        else: # Concise summary
            map_prompt_template = """
            Write a concise summary of the following video transcript segments:
            
            "{segments}"
            """
            
            combine_prompt_template = """Write a concise summary of the following video transcript that combines the previous summaries
            "{transcript}"
            """
        
        map_chain = PromptTemplate(template=map_prompt_template, input_variables=["segments"])
        combine_chain = PromptTemplate(template=combine_prompt_template, input_variables=["transcript"])

        chain = map_chain | self.client.llm | combine_chain | self.client.llm

        response = chain.invoke({"segments": documents})

        return response.content

    def generate_response(self, question: str) -> str:
        """
        Generate a response to a question about the article.

        Args:
            question (str): The question to generate a response to.
        """
        response = self.client.qa_chain.invoke({"question": question})
        return response["answer"]
        