from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime


class GeneratedResponse(BaseModel):
    """
    GeneratedResponse entity representing the chatbot's response to a query
    """
    id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: Optional[List[Dict[str, Any]]] = Field(default=[])
    conversation_context: Optional[Dict[str, Any]] = Field(default={})
    status: str = "DRAFT"  # DRAFT -> GENERATED -> REVIEWED -> PUBLISHED
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }