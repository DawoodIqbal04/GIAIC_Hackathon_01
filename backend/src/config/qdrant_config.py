"""
Qdrant configuration module for the Book RAG Chatbot
"""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Optional
from .settings import settings


def get_qdrant_client() -> QdrantClient:
    """
    Get a configured Qdrant client instance
    """
    if settings.qdrant_api_key:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=True
        )
    else:
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port
        )
    
    return client


def initialize_qdrant_collection(client: QdrantClient, collection_name: Optional[str] = None):
    """
    Initialize the Qdrant collection for storing document embeddings
    """
    collection_name = collection_name or settings.qdrant_collection_name
    
    # Check if collection exists
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists")
    except:
        # Create collection if it doesn't exist
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # Default size for OpenAI embeddings
                distance=models.Distance.COSINE
            )
        )
        print(f"Created collection '{collection_name}'")


def get_qdrant_config():
    """
    Get Qdrant configuration parameters
    """
    return {
        "host": settings.qdrant_host,
        "port": settings.qdrant_port,
        "collection_name": settings.qdrant_collection_name,
        "api_key": settings.qdrant_api_key
    }