from typing import Dict

from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseLanguageModel

from src.states.blog import Blog, BlogState

class BlogNode:
    """
    A class tp represent the blog node
    """
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        
    def title_creation(self, state: BlogState) -> Dict:
        """
        Create the title for the blog
        """
        if "topic" in state and state["topic"]:
            prompt = """
                You are an expert blog content writer. Use Markdown formatting. Generate
                a blog title for the {topic}. This title should be creative and SEO friendly
            """
            system_message = prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            
            return {"blog": Blog(title=response.content)}
        return {"blog": Blog()}
        
    def content_generation(self, state: BlogState) -> Dict:
        if "topic" in state and state["topic"]:
            system_prompt = """
                You are expert blog writer. Use Markdown formatting.
                Generate a detailed blog content with detailed breakdown for the {topic}
            """
            system_message = system_prompt.format(topic=state["topic"])
            response = self.llm.invoke(system_message)
            
            return {"blog": Blog(title=state["blog"].title, content=response.content)}
        return {"blog": Blog()}
    
    def verify_language(self, state: BlogState) -> Dict:
        """
        Verify the provided language is a valid or recognized human language.
        """
        language_prompt = """
            Your task is to determine whether the provided language is a valid, recognized human language.
            Follow these rules:
                1.	A valid language must be a real human language that is currently or historically used for communication (e.g., English, Yoruba, Mandarin, Latin).
                2.	It may include dialects or standardized variants (e.g., Brazilian Portuguese, Swiss German).
                3.	It must not be:
                    - a fictional or constructed language (e.g., Klingon, Elvish)
                    - an abbreviation that does not name a language (e.g., “lax,” “eng lang”)
                    - a code unrelated to language (e.g., airport codes, product names)
                4.	Output only one of the following:
                    - “Valid language”
                    - “Invalid language”

            LANGUAGE TO CHECK:
            {language}
        """
        language = state["language"]
        messages = [
            HumanMessage(language_prompt.format(language=language))
        ]
        response = self.llm.invoke(messages)

        return {"language_check": response.content}
    
    def route(self, state: BlogState) -> str:
        return state["language_check"]
        
    def translation(self, state: BlogState) -> Dict:
        """
        Translate the content to the specified language.
        """
        translation_prompt = """
            Translate the following content into {language}.
            - If the provided language is not a valid or recognized human language, simply say “The provided language is incorrect.”
            - Maintain the original tone, style, and formatting.
            - Adapt cultural references and idioms to be appropriate for {language}.

            ORIGINAL CONTENT:
            {blog_content}
        """
        blog_content = state["blog"].content
        language = state["language"]
        messages = [
            HumanMessage(translation_prompt.format(language=language, blog_content=blog_content))
        ]
        response = self.llm.with_structured_output(Blog).invoke(messages)
        
        return {
            "blog": Blog(
                title=state["blog"].title,
                content=blog_content,
                translated_title=response.title,
                translated_content=response.content
            )
        }
