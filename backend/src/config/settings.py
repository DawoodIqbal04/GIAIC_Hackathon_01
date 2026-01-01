from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    # API Settings
    api_title: str = "Book RAG Chatbot API"
    api_version: str = "1.0.0"
    api_description: str = "API for the Book RAG Chatbot that answers questions about book content"

    # OpenAI Settings
    openai_api_key: str
    openai_model: str = "gpt-3.5-turbo"

    # Qdrant Settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "book_content"

    # Database Settings
    database_url: str  # Neon Postgres connection string

    # CORS Settings
    cors_origins: list = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()