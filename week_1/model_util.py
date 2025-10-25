from main import LLMChatApp

def is_groq_model(model: str) -> bool:
    """
    Check if the given model name is a Groq model.
    
    Parameters:
        - model (str) : The name of the model to check.
    """
    
    return model in LLMChatApp.SUPPORTED_GROQ_MODELS