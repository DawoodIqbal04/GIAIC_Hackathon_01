"""
PostgreSQL configuration module for the Book RAG Chatbot
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from .settings import settings


def get_database_url():
    """
    Get the database URL from settings
    """
    return settings.database_url


def get_sync_engine():
    """
    Get a synchronous database engine
    """
    return create_engine(settings.database_url)


def get_async_engine():
    """
    Get an asynchronous database engine
    """
    return create_async_engine(settings.database_url)


def get_session_maker(async_engine=None):
    """
    Get a session maker for database operations
    """
    if async_engine:
        return sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)
    else:
        async_engine = get_async_engine()
        return sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


def get_postgres_config():
    """
    Get PostgreSQL configuration parameters
    """
    return {
        "database_url": settings.database_url
    }