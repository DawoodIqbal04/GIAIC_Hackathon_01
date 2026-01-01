from typing import Optional
from pydantic import BaseModel, validator
from uuid import UUID


def validate_uuid(value: str) -> UUID:
    """
    Validate that a string is a valid UUID
    """
    try:
        return UUID(value)
    except ValueError:
        raise ValueError(f"Invalid UUID: {value}")


def validate_document_content_length(content: str) -> str:
    """
    Validate document content length (max 10,000 characters)
    """
    if len(content) > 10000:
        raise ValueError("Document content exceeds maximum length of 10,000 characters")
    return content


def validate_relevance_score(score: float) -> float:
    """
    Validate relevance score is between 0 and 1
    """
    if score < 0.0 or score > 1.0:
        raise ValueError("Relevance score must be between 0.0 and 1.0")
    return score


def validate_query_content_length(content: str) -> str:
    """
    Validate query content length (between 10 and 5000 characters)
    """
    if len(content) < 10:
        raise ValueError("Query content is too short, minimum 10 characters required")
    if len(content) > 5000:
        raise ValueError("Query content is too long, maximum 5000 characters allowed")
    return content


def validate_required_fields(data: dict, required_fields: list) -> bool:
    """
    Validate that required fields are present in a data dictionary
    """
    for field in required_fields:
        if field not in data or data[field] is None:
            raise ValueError(f"Required field '{field}' is missing or None")
    return True


def is_valid_json(json_str: str) -> bool:
    """
    Check if a string is valid JSON
    """
    import json
    try:
        json.loads(json_str)
        return True
    except ValueError:
        return False