import os

from langchain_openai import ChatOpenAI

class OpenAIModel:
    def __init__(self, user_controls):
        self.user_controls = user_controls
        
    def get_model(self) -> ChatOpenAI:
        try:
            os.environ["OPENAI_API_KEY"] = self.user_controls["OPENAI_API_KEY"]
            llm = ChatOpenAI(model=self.user_controls["openai_model"])
            
            return llm
        except Exception as e:
            raise ValueError("Error occured with exception: {e}")
            