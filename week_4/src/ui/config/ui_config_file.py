import sys
import os
from configparser import ConfigParser

class Config:
    # Get the config file path with absolute path
    def __init__(self, config_file=os.path.join(os.path.dirname(__file__), "ui_config_file.ini")):
        self.config = ConfigParser()
        self.config.read(config_file)
        
    def get_llm_options(self):
        LLM = self.config["DEFAULT"].get("LLM_OPTIONS")
        if LLM:
            return LLM.split(", ")
        return ""
    
    def get_usecase_options(self):
        USECASE = self.config["DEFAULT"].get("USECASE_OPTIONS")
        if USECASE:
            return USECASE.split(", ")
        return ""

    def get_openai_model_options(self):
        OPENAI_MODEL = self.config["DEFAULT"].get("OPENAI_MODEL_OPTIONS")
        if OPENAI_MODEL:
            return OPENAI_MODEL.split(", ")
        return ""
    
    def get_groq_model_options(self):
        GROQ_MODEL = self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS")
        if GROQ_MODEL:
            return GROQ_MODEL.split(", ")
        return ""
    
    def get_page_title(self):
        PAGE_TITLE = self.config["DEFAULT"].get("PAGE_TITLE")
        if PAGE_TITLE:
            return PAGE_TITLE
        return ""
        