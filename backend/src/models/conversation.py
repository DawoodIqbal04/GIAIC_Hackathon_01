from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime


class Conversation(BaseModel):
    """
    Conversation entity representing a user's conversation session
    """
    id: UUID = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None
    session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default={})
    is_active: bool = True
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }