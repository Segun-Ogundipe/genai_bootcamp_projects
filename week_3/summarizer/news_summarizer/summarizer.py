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

class NewsSummarizer:
    def __init__(
        self,
        llm_provider: str=SUPPORTED_LLM_PROVIDERS[0],
        llm_name: str=SUPPORTED_OPENAI_MODELS[0],
        llm_api_key: Optional[str]=None,
        embedding_provider: str=SUPPORTED_EMBEDDING_PROVIDERS[0],
        embedding_model_name: str=SUPPORTED_OPENAI_EMBEDDING_MODELS[0],
        embedding_api_key: Optional[str]=None,
    ):
        """
        Initialize the NewsSummarizer with choice of model.

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
            collection_name="news-store",
            embedding_provider=self.embedding_provider,
            embedding_model_name=self.embedding_model_name,
            embedding_api_key=self.embedding_api_key,
        )
        
        self.client = LLMClient(
            provider=self.llm_provider,
            model_name=self.llm_name,
            api_key=self.llm_api_key,
            store=self.store,
        )

    def download_and_process_article(self, url: str) -> List[Document]:
        """
        Download and process a news article from a URL.

        Args:
            url (str): The URL of the news article to download and process.
        """
        documents = self.store.load_document(source=url, source_type="news")

        try:
            processed_documents = self.store.add_to_store(documents)
        except Exception:
            processed_documents = self.store.create_store(documents)

        return processed_documents

    def summarize_article(self, url: str, summary_type: str="concise") -> str:
        """
        Summarize a news article from a URL.

        Args:
            url (str): The URL of the news article to summarize.
            summary_type (str): The type of summary to generate ("detailed" or "concise"). Defaults to "concise".
        """
        documents = self.download_and_process_article(url)

        if summary_type == "detailed":
            map_prompt_template = """
            Write a detailed summary of the following news article segments:
            
            "{segments}"
            """
            
            combine_prompt_template = """
            Write a detailed summary of the following news article that combines the previous summaries:
            
            "{article}"
            """
        else: # Concise summary
            map_prompt_template = """
            Write a concise summary of the following news article segments:
            
            "{segments}"
            """
            
            combine_prompt_template = """Write a concise summary of the following text that combines the previous summaries
            "{article}"
            """

        map_chain = PromptTemplate(template=map_prompt_template, input_variables=["segments"])
        combine_chain = PromptTemplate(template=combine_prompt_template, input_variables=["article"])

        chain = map_chain | self.client.llm | combine_chain | self.client.llm

        response = chain.invoke({"segments": documents})

        return response.content
