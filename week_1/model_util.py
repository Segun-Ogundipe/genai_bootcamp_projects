SUPPORTED_GROQ_MODELS = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
]

SUPPORTED_OPENAI_MODELS = [
        "gpt-4o-mini",
        "gpt-5-2025-08-07",
        "gpt-5-mini-2025-08-07",
        "gpt-5-nano-2025-08-07"
]

def is_groq_model(model: str) -> bool:
    """
    Check if the given model name is a Groq model.
    
    Parameters:
        - model (str) : The name of the model to check.
    """
    
    return model in SUPPORTED_GROQ_MODELS