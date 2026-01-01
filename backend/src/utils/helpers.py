import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime


def generate_uuid() -> str:
    """
    Generate a UUID string
    """
    return str(uuid.uuid4())


def get_current_timestamp() -> str:
    """
    Get current timestamp in ISO format
    """
    return datetime.utcnow().isoformat()


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if truncated
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_source_reference(title: str, section: Optional[str] = None, 
                          page: Optional[int] = None) -> str:
    """
    Format a source reference for citations
    """
    ref_parts = [title]
    if section:
        ref_parts.append(f"Section: {section}")
    if page:
        ref_parts.append(f"Page: {page}")
    
    return ", ".join(ref_parts)


def calculate_similarity_score(text1: str, text2: str) -> float:
    """
    Calculate a basic similarity score between two texts
    This is a simplified implementation - in practice, you'd use embeddings
    """
    if not text1 or not text2:
        return 0.0
    
    # Simple word overlap as a placeholder
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    if not union:
        return 0.0
    
    return len(intersection) / len(union)


def validate_query_content(content: str) -> bool:
    """
    Validate query content meets requirements (10-5000 chars)
    """
    if not content:
        return False
    
    if len(content) < 10 or len(content) > 5000:
        return False
    
    return True