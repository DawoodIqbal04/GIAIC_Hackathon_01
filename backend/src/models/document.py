from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime


class Document(BaseModel):
    """
    Document entity representing book content that can be retrieved
    """
    id: UUID = Field(default_factory=uuid4)
    title: str
    content: str = Field(..., max_length=10000)  # Max 10k chars per chunk
    source_path: str
    embedding: Optional[str] = None  # Vector representation as string
    metadata: Optional[Dict[str, Any]] = Field(default={})
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }