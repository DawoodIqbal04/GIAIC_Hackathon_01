from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Import settings
from ..config.settings import settings

# Import routes
from .routes import chat, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown events
    """
    # Startup
    print("Starting up Book RAG Chatbot API...")
    
    # Initialize services, database connections, etc. here
    # For example: initialize_qdrant_client(), initialize_postgres_connection()
    
    yield
    
    # Shutdown
    print("Shutting down Book RAG Chatbot API...")
    # Cleanup resources here


# Create FastAPI app instance
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose headers for client access
    expose_headers=["Access-Control-Allow-Origin"]
)

# Include API routes
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(health.router, prefix="/health", tags=["health"])

# Add basic logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )