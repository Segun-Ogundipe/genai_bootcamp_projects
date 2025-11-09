from typing import List

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from newspaper import Article

class ArticleLoader(BaseLoader):
    def __init__(self, url: str):
        """
        Initialize the ArticleLoader.

        Args:
            url (str): The URL of the article to load.
        """
        self.url = url

    def load(self) -> List[Document]:
        """
        Download and parse the article into document objects.

        Returns:
            List[Document]: A list of Document objects representing the article.
        """
        try:
            # Download and parse the article
            article = Article(self.url)
            article.download()
            article.parse()

            return [
                Document(page_content=article.text, metadata={
                    "title": article.title,
                    "url": article.url,
                    "authors": article.authors,
                    "published_date": article.publish_date,
                    "keywords": article.keywords,
                    "summary": article.summary
                })
            ]
        except Exception as e:
            raise RuntimeError(f"Error downloading and parsing the article: {e}")

