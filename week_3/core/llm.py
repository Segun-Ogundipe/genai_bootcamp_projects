from typing import Optional

from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSequence
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config.settings import env_config
from utils.model_util import SUPPORTED_GROQ_MODELS, SUPPORTED_OPENAI_MODELS

class LLMClient:
    def __init__(
        self,
        provider: str="openai",
        model_name: str=SUPPORTED_OPENAI_MODELS[0],
        api_key: Optional[str]=None,
        store: Chroma=None
    ):
        """
        Initialize the LLMClient.

        Args:
            provider (str): The provider to use for the LLM. Defaults to "openai" ("openai" or "groq").
            model_name (str): The name of the model to use for the LLM. Defaults to "gpt-5-nano-2025-08-07"
            system_prompt (str): The system prompt to use for the LLM. Defaults to "You are a helpful assistant."
            api_key (Optional[str]): Provider API key (falls back to environment variable if not provided).
        """
        self.provider = provider
        self.model_name = model_name
        self.store = store

        self.__api_key = api_key or self.__get_api_key()
        self.llm = self.__initialize_llm()
        self.qa_chain = self.__create_qa_chain()

    def __get_api_key(self) -> str:
        """Get API key from environment variables"""
        if self.model_name in SUPPORTED_GROQ_MODELS:
            api_key = env_config.groq_api_key
        elif self.model_name in SUPPORTED_OPENAI_MODELS:
            api_key = env_config.openai_api_key
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        if not api_key:
            raise ValueError(f"API key is not set for the selected model: {self.model_name}")

        return api_key

    def __initialize_llm(self) -> BaseChatModel:
        """Create the LLM instance based on the provider and model_name"""
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.__api_key,
                temperature=0.2,
            )
        elif self.provider == "groq":
            return ChatGroq(
                model=self.model_name,
                api_key=self.__api_key,
                temperature=0.2
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def __create_qa_chain(self) -> RunnableSequence:
        """Create the QA chain with prompt template"""
        memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.store.as_retriever(),
            memory=memory
        )
