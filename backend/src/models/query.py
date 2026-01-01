from pydantic import BaseModel, Field, validator
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime


class Query(BaseModel):
    """
    Query entity representing a user's natural language query about book content
    """
    id: UUID = Field(default_factory=uuid4)
    content: str = Field(..., min_length=10, max_length=5000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    selected_text: Optional[str] = None
    status: str = "PENDING"  # PENDING -> PROCESSING -> COMPLETED

    @validator('content')
    def validate_content_length(cls, v):
        if len(v) < 10:
            raise ValueError('Query content must be at least 10 characters long')
        if len(v) > 5000:
            raise ValueError('Query content must not exceed 5000 characters')
        return v

    @validator('selected_text')
    def validate_selected_text_length(cls, v):
        if v and len(v) > 10000:
            raise ValueError('Selected text must not exceed 10000 characters')
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }