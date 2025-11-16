from typing import Optional, TypedDict
from pydantic import BaseModel, Field

class Blog(BaseModel):
    title: Optional[str] = Field(description="The original title of the blog post", default="")
    content: Optional[str] = Field(description="The original content of the blog post", default="")
    translated_title: Optional[str] = Field(description="The translated title of the blog post", default="")
    translated_content: Optional[str] = Field(description="The translated content of the blog post", default="")
    
class BlogState(TypedDict):
    topic: str
    blog: Blog
    language: str
    language_check: str
