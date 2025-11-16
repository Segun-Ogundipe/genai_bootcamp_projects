from langchain_core.language_models import BaseLanguageModel
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph

from src.nodes.blog import BlogNode
from src.states.blog import BlogState

class GraphBuilder:
    def __init__(self, llm: BaseLanguageModel) -> None:
        self.llm = llm
        self.graph = StateGraph(BlogState)
        
    def build_topic_graph(self) -> StateGraph:
        """
        Build a graph to generate blog based on topic
        """
        self.blog_node = BlogNode(self.llm)
        
        # Nodes
        self.graph.add_node("title_creation", self.blog_node.title_creation)
        self.graph.add_node("content_generation", self.blog_node.content_generation)
        
        # Edges
        self.graph.add_edge(START, "title_creation")
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", END)
        
        return self.graph
    
    def build_language_graph(self) -> StateGraph:
        """
        Build a graph for blog generation with inputs topic and language
        """
        self.blog_node = BlogNode(self.llm)
        
        # Nodes
        self.graph.add_node("verify_language", self.blog_node.verify_language)
        self.graph.add_node("title_creation", self.blog_node.title_creation)
        self.graph.add_node("content_generation", self.blog_node.content_generation)
        self.graph.add_node("translation", self.blog_node.translation)
        
        # Edges
        self.graph.add_edge(START, "verify_language")
        self.graph.add_conditional_edges(
            "verify_language",
            self.blog_node.route,
            {
                "Valid language": "title_creation",
                "Invalid language": END
            }
        )
        self.graph.add_edge("title_creation", "content_generation")
        self.graph.add_edge("content_generation", "translation")
        self.graph.add_edge("translation", END)
        
        return self.graph
    
    def setup_graph(self, usecase) -> CompiledStateGraph:
        """
        Setup the graph based on the use case
        """
        try:
            if usecase == "Blog Generation":
                self.build_topic_graph()
            if usecase == "Blog Generation with Language Translation":
                self.build_language_graph()
                
            return self.graph.compile()
        except Exception as e:
            raise ValueError(f"Error: Graph setup failed- {e}")
