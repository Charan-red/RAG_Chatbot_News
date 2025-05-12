<<<<<<< HEAD
import os
import json
import glob
from typing import List, Dict, Any
from datetime import datetime
from huggingface_hub import login
import faiss
import numpy as np

from llama_index.core import Document
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
# from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.core import HuggingFaceEmbedding
# from llama_index.core import HuggingFaceLLM
from transformers import AutoModelForSeq2SeqLM
from llama_index.core import Settings
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
# from IPython.display import Markdown, display 


class NewsRAGSystem:
    def __init__(self, 
                 model_name="google/flan-t5-base", 
                 embedding_model="google/flan-t5-base",
                 index_path="./news_index",
                 device="cpu",
                 access_token='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'
                 ):
        """
        Initialize the RAG system with a model
        
        Args:
            model_name: HuggingFace model name for LLM
            embedding_model: Model for creating embeddings
            index_path: Path to store the vector index
            device: Device to run models on ('cpu' or 'cuda')
        """
        login(token=access_token)
        self.index_path = index_path
        self.device = device
        self.model_name = model_name
        self.access_token = access_token
        
        # Step 1: Set up embedding model FIRST
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            device=device,
            max_length=512
        )
        
        # Step 2: Determine the embedding dimension by creating a test embedding
        self.embed_dimension = self._get_embedding_dimension()
        print(f"Detected embedding dimension: {self.embed_dimension}")
        
        # Step 3: Set up LLM
        self.llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            device_map=device,
            model=AutoModelForSeq2SeqLM.from_pretrained(model_name),
            model_kwargs={"temperature": 0.7, "max_length": 2048}
        )
        
        # Step 4: Update the global Settings object with our models
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        # Initialize or load index
        self._initialize_index()
    
    def _get_embedding_dimension(self):
        """Determine the embedding dimension by creating a test embedding"""
        test_text = "This is a test sentence to determine embedding dimension."
        embedding = self.embed_model.get_text_embedding(test_text)
        return len(embedding)
        
    def _initialize_index(self):
        """Initialize or load the vector index"""
        if os.path.exists(f"{self.index_path}/faiss_index"):
            print("Loading existing index...")
            try:
                # Load existing FAISS index from disk using the corrected API
                vector_store = FaissVectorStore.from_persist_dir(f"{self.index_path}/faiss_index")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Use load_index_from_storage instead of VectorStoreIndex.from_storage_context
                self.index = load_index_from_storage(storage_context)
                print("Successfully loaded index")
            except Exception as e:
                print(f"Error loading index: {str(e)}")
                print("Will create a new index instead")
                self.index = None
        else:
            print("No existing index found. Will create when documents are added.")
            self.index = None
    
    def add_articles(self, articles: List[Dict[str, Any]]):
        """
        Add articles to the index
        
        Args:
            articles: List of article dictionaries
        """
        if not articles:
            print("No articles to add.")
            return
        
        # Convert articles to documents
        documents = []
        for article in articles:
            # Create metadata with all article info except content
            metadata = {k: v for k, v in article.items() if k != 'content'}
            
            # Add source identifier to help with attribution
            text = f"SOURCE: {article['source']}\n"
            text += f"TITLE: {article['title']}\n"
            text += f"DATE: {article.get('date', 'Unknown date')}\n\n"
            text += article['content']
            
            document = Document(
                text=text,
                metadata=metadata
            )
            documents.append(document)
        
        # Create or update index
        if self.index is None:
            try:
                # Create a new FAISS index with the correct dimension
                faiss_index = faiss.IndexFlatL2(self.embed_dimension)
                
                # Create vector store with the FAISS index
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                
                # Create index from documents
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context
                )
                
                # Create directory if it doesn't exist
                os.makedirs(f"{self.index_path}/faiss_index", exist_ok=True)
                
                # Persist index
                self.index.storage_context.persist(persist_dir=f"{self.index_path}/faiss_index")
                print("Successfully created and persisted new index")
            except Exception as e:
                print(f"Error creating index: {str(e)}")
                raise
        else:
            try:
                # Add documents to existing index
                for document in documents:
                    self.index.insert(document)
                
                # Persist updated index
                self.index.storage_context.persist(persist_dir=f"{self.index_path}/faiss_index")
                print("Successfully updated and persisted index")
            except Exception as e:
                print(f"Error updating index: {str(e)}")
                raise
            
        print(f"Added {len(documents)} documents to index")
    
    def load_from_json_files(self, directory="./data"):
        """
        Load articles from JSON files in a directory
        
        Args:
            directory: Directory containing news article JSON files
        """
        json_files = glob.glob(f"{directory}/news_articles_*.json")
        
        if not json_files:
            print(f"No news article files found in {directory}")
            return
        
        try:
            # Sort by modification time to get the latest first
            latest_file = sorted(json_files, key=os.path.getmtime, reverse=True)[0]
            print(f"Loading articles from {latest_file}")
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                
            self.add_articles(articles)
        except Exception as e:
            print(f"Error loading articles: {str(e)}")
            raise
        
    def query(self, question: str, num_results: int = 3):
        """
        Query the RAG system
        
        Args:
            question: User's question about news
            num_results: Number of source documents to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.index is None:
            return {
                "answer": "No news articles have been indexed yet. Please add articles first.",
                "sources": []
            }
        
        try:
            # Create query engine with citation information
            query_engine = self.index.as_query_engine(
                similarity_top_k=num_results,
                response_mode="tree_summarize"
            )
            
            # Get response
            response = query_engine.query(question)
            
            # Extract source nodes
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    if node.node.metadata:
                        sources.append({
                            "source": node.node.metadata.get("source", "Unknown"),
                            "title": node.node.metadata.get("title", "Unknown title"),
                            "url": node.node.metadata.get("url", "#"),
                            "date": node.node.metadata.get("date", "Unknown date"),
                            "score": node.score
                        })
            
            return {
                "answer": response.response,
                "sources": sources
            }
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "sources": []
            }
=======
import os
import json
import glob
from typing import List, Dict, Any
from datetime import datetime

from llama_index import VectorStoreIndex, Document, ServiceContext, StorageContext
from llama_index.vector_stores import FAISSVectorStore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import HuggingFaceLLM

class NewsRAGSystem:
    def __init__(self, 
                 model_name="meta-llama/Llama-2-7b-chat-hf", 
                 embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                 index_path="./news_index",
                 device="cpu"):
        """
        Initialize the RAG system with a Llama model
        
        Args:
            model_name: HuggingFace model name for LLM (default: Llama-2-7b-chat-hf)
            embedding_model: Model for creating embeddings
            index_path: Path to store the vector index
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.index_path = index_path
        self.device = device
        self.model_name = model_name
        
        # Set up embedding model
        self.embed_model = HuggingFaceEmbedding(
            model_name=embedding_model,
            device=device
        )
        
        # Set up LLM
        self.llm = HuggingFaceLLM(
            model_name=model_name,
            tokenizer_name=model_name,
            device_map=device,
            model_kwargs={"temperature": 0.7, "max_length": 2048}
        )
        
        # Set up service context
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model
        )
        
        # Initialize or load index
        self._initialize_index()
        
    def _initialize_index(self):
        """Initialize or load the vector index"""
        if os.path.exists(f"{self.index_path}/faiss_index"):
            print("Loading existing index...")
            vector_store = FAISSVectorStore.from_persist_dir(f"{self.index_path}/faiss_index")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex.from_storage_context(
                storage_context=storage_context,
                service_context=self.service_context
            )
        else:
            print("No existing index found. Will create when documents are added.")
            self.index = None
    
    def add_articles(self, articles: List[Dict[str, Any]]):
        """
        Add articles to the index
        
        Args:
            articles: List of article dictionaries
        """
        if not articles:
            print("No articles to add.")
            return
        
        # Convert articles to documents
        documents = []
        for article in articles:
            # Create metadata with all article info except content
            metadata = {k: v for k, v in article.items() if k != 'content'}
            
            # Add source identifier to help with attribution
            text = f"SOURCE: {article['source']}\n"
            text += f"TITLE: {article['title']}\n"
            text += f"DATE: {article.get('date', 'Unknown date')}\n\n"
            text += article['content']
            
            document = Document(
                text=text,
                metadata=metadata
            )
            documents.append(document)
        
        # Create or update index
        if self.index is None:
            # Create new vector store and index
            vector_store = FAISSVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                service_context=self.service_context
            )
            
            # Create directory if it doesn't exist
            os.makedirs(f"{self.index_path}/faiss_index", exist_ok=True)
            
            # Persist index
            self.index.storage_context.persist(persist_dir=f"{self.index_path}/faiss_index")
        else:
            # Add documents to existing index
            for document in documents:
                self.index.insert(document)
            
            # Persist updated index
            self.index.storage_context.persist(persist_dir=f"{self.index_path}/faiss_index")
            
        print(f"Added {len(documents)} documents to index")
    
    def load_from_json_files(self, directory="./data"):
        """
        Load articles from JSON files in a directory
        
        Args:
            directory: Directory containing news article JSON files
        """
        json_files = glob.glob(f"{directory}/news_articles_*.json")
        
        if not json_files:
            print(f"No news article files found in {directory}")
            return
        
        # Sort by modification time to get the latest first
        latest_file = sorted(json_files, key=os.path.getmtime, reverse=True)[0]
        print(f"Loading articles from {latest_file}")
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
            
        self.add_articles(articles)
        
    def query(self, question: str, num_results: int = 3):
        """
        Query the RAG system
        
        Args:
            question: User's question about news
            num_results: Number of source documents to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        if self.index is None:
            return {
                "answer": "No news articles have been indexed yet. Please add articles first.",
                "sources": []
            }
        
        # Create query engine with citation information
        query_engine = self.index.as_query_engine(
            similarity_top_k=num_results,
            service_context=self.service_context,
            response_mode="tree_summarize"
        )
        
        # Get response
        response = query_engine.query(question)
        
        # Extract source nodes
        sources = []
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                if node.node.metadata:
                    sources.append({
                        "source": node.node.metadata.get("source", "Unknown"),
                        "title": node.node.metadata.get("title", "Unknown title"),
                        "url": node.node.metadata.get("url", "#"),
                        "date": node.node.metadata.get("date", "Unknown date"),
                        "score": node.score
                    })
        
        return {
            "answer": response.response,
            "sources": sources
        }

# Example usage
if __name__ == "__main__":
    rag_system = NewsRAGSystem()
    
    # Load articles from JSON files
    rag_system.load_from_json_files()
    
    # Example query
    result = rag_system.query("What's the latest news about climate change?")
    print("\nAnswer:", result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['title']} ({source['source']})")
>>>>>>> 092e1a8a384a3aacc6591d60604c63f1d688b818
