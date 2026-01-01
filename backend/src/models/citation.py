from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID, uuid4
from datetime import datetime


class Citation(BaseModel):
    """
    Citation entity representing references to sources in generated responses
    """
    id: UUID = Field(default_factory=uuid4)
    response_id: UUID
    document_id: UUID
    source_reference: str  # Specific reference like chapter, section, page
    content_snippet: str  # The relevant snippet from the source
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v)
        }