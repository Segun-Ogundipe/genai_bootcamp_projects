import sys
from typing import Union
from groq import Groq
from openai import OpenAI
from app_config import env_config
from model_util import is_groq_model, SUPPORTED_GROQ_MODELS, SUPPORTED_OPENAI_MODELS


class LLMChatApp:
    """Main application class to interact with Groq and OpenAI APIs."""
    
    def __init__(
        self,
        model: str=SUPPORTED_GROQ_MODELS[0],
        model_name: Union[str, None]=None,
        system_prompt: Union[str, None]=None,
        api_key: Union[str, None]=None
    ):
        """
        Initialize the LLM Application.

        Parameters:
            - model (str, optional) : The model to use for generating completions. Defaults to `llama-3.1-8b-instant`.
            - api_key (str, optional) : The API key for authentication.
                - Use a **Groq API key** when using Groq models.
                - Use an **OpenAI API key** when using OpenAI models.
                If not provided, the key is automatically read from the corresponding environment variable — `GROQ_API_KEY` or `OPENAI_API_KEY`, depending on the selected model.
        """
        self.__model = model
        
        if self.__model in SUPPORTED_GROQ_MODELS:
            self.__api_key = api_key or env_config.groq_api_key
            
            if not self.__api_key:
                raise ValueError("Groq API key is required for the selected Groq model. Provide it via the `api_key` parameter or set the `GROQ_API_KEY` environment variable.")
            
            self.__client = Groq(api_key=self.__api_key)
            
        elif self.__model in SUPPORTED_OPENAI_MODELS:
            self.__api_key = api_key or env_config.openai_api_key
            
            if not self.__api_key:
                raise ValueError("OpenAI API key is required for the selected OpenAI model. Provide it via the `api_key` parameter or set the `OPENAI_API_KEY` environment variable.")
            
            self.__client = OpenAI(api_key=self.__api_key)
        else:
            raise ValueError(f"Unsupported model: {self.__model}. Please choose a supported Groq or OpenAI model.")
            
        self.__conversation_history = []
        self.__set_model_name(model_name=model_name)
        self.__set_system_prompt(system_prompt=system_prompt)
        
    def chat(self, user_message: str, temperature: float=0.5, max_tokens: int=1024) -> str | None:
        """
        Engage in a chat with the LLM.

        Parameters:
            - user_message (str) : The user's message.
            - system_prompt (str) : The system prompt to set context.
            - temperature (float, optional) : Sampling temperature for response generation (0-2). Defaults to 0.5.
            - max_tokens (int, optional) : Maximum number of tokens in the response. Defaults to 1024.

        Returns:
            (str | None) : The assistant's response or None if an error occurs.
        """
        # Add current user's message to the conversation history
        self.__conversation_history.append({"role": "user", "content": user_message})
        
        # Add system prompt at the beginning of conversation history 
        messages: list = [{"role": "system", "content": self.get_system_prompt()}] + self.get_history()
        
        try:
            # Make LLM call
            params = {
                "model": self.__model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if is_groq_model(self.__model):
                params["max_tokens"] = max_tokens
            else:
                params["max_completion_tokens"] = max_tokens

            response = self.__client.chat.completions.create(**params)
            
            # Extract response text
            assistant_message = response.choices[0].message.content
            self.__conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
        except Exception as e:
            print(f"Error during chat completion: {e}")
            raise e
        
    def get_api_key(self) -> str | None:
        """
        Get the API key being used.
        
        Returns:
            (str) : The API key.
        """
        return self.__api_key
        
    def clear_history(self):
        """Clear the conversation history."""
        self.__conversation_history = []
    
    def get_history(self):
        """Retrieve the conversation history."""
        return self.__conversation_history
    
    def get_model_name(self) -> str:
        """
        Get the name of the model.
        
        Returns:
            (str) : The name of the model.
        """
        return self.__model_name
    
    def __set_model_name(self, model_name: Union[str, None]):
        """
        Set a new model name.
        
        Parameters:
            - model_name (str) : The new model name to set.
        """
        self.__model_name = model_name if model_name else "Tidal"
        
    def get_system_prompt(self) -> str:
        """
        Get the current system prompt.
        
        Returns:
            (str | None) : The current system prompt.
        """
        return self.__system_prompt
    
    def __set_system_prompt(self, system_prompt: Union[str, None]):
        """
        Set a new system prompt.
        
        Parameters:
            - system_prompt (str) : The new system prompt to set.
        """
        self.__system_prompt = f"You are {self.__model_name}. {system_prompt if system_prompt else 'A helpful assistant.'}"
        

if __name__ == "__main__":
    EXIT_MESSAGE = "\nExiting the chat. Goodbye!"
    EXIT_COMMANDS = ["exit", "quit"]
    
    print("\nWelcome to A Basic LLM Chat Application!")
    print("Type 'exit' or 'quit' to end the chat.")
    
    try:
        model = input(f"\nEnter the model. Press Enter to skip (default: {SUPPORTED_GROQ_MODELS[0]}): ").strip()
        if model.lower() in EXIT_COMMANDS:
            print(EXIT_MESSAGE)
            sys.exit(0)
        
        model_name = input("\nEnter the model name. Press Enter to skip (default: Tidal): ").strip()
        if model_name.lower() in EXIT_COMMANDS:
            print(EXIT_MESSAGE)
            sys.exit(0)
            
        system_prompt = input("\nEnter a system prompt. Press Enter to skip (optional): ").strip()
        if system_prompt.lower() in EXIT_COMMANDS:
            print(EXIT_MESSAGE)
            sys.exit(0)
            
        api_key = input("\nEnter an API key. Press Enter to skip (optional): ").strip()
        if api_key.lower() in EXIT_COMMANDS:
            print(EXIT_MESSAGE)
            sys.exit(0)

        app = LLMChatApp(
            model=model or f"{SUPPORTED_GROQ_MODELS[0]}",
            model_name=model_name or "Tidal",
            system_prompt=system_prompt or None,
            api_key=api_key
        )
        
        while True:
            message = input("\nAsk me anything: ").strip()
            if message.lower() in EXIT_COMMANDS:
                print(EXIT_MESSAGE)
                break
            
            response = app.chat(user_message=message)
            print(f"\nAssistant response: {response}")
    except KeyboardInterrupt:
        print(EXIT_MESSAGE)
        sys.exit(0)
        