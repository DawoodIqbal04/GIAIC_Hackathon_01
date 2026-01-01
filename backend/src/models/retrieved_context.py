from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime


class RetrievedContext(BaseModel):
    """
    RetrievedContext entity representing relevant portions of documents retrieved for a query
    """
    id: UUID = Field(default_factory=uuid4)
    query_id: UUID
    document_id: UUID
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)  # Between 0 and 1
    source_citation: str
    status: str = "CREATED"  # CREATED -> RANKED -> SELECTED
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }