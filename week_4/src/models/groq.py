import os

from langchain_groq import ChatGroq

class GroqModel:
    def __init__(self, user_controls):
        self.user_controls = user_controls
        
    def get_model(self) -> ChatGroq:
        try:
            os.environ["GROQ_API_KEY"] = self.user_controls["GROQ_API_KEY"]
            llm = ChatGroq(model=self.user_controls["groq_model"])
            
            return llm
        except Exception as e:
            raise ValueError("Error occured with exception: {e}")
            