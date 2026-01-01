from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from uuid import UUID
import json

from ...models.query import Query
from ...models.conversation import Conversation
from ...services.rag_service import RAGService
from ...config.settings import settings

router = APIRouter()
rag_service = RAGService()


@router.post("/query")
async def chat_query(
    query: str,
    conversation_id: Optional[str] = None,
    selected_text: Optional[str] = None,
    include_citations: bool = True
):
    """
    Submit a query to the RAG chatbot and receive a response
    """
    try:
        # Validate conversation_id if provided
        conversation_uuid = None
        if conversation_id:
            try:
                conversation_uuid = UUID(conversation_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid conversation ID format")
        
        # Create a Query object
        query_obj = Query(
            content=query,
            conversation_id=conversation_uuid,
            selected_text=selected_text
        )
        
        # Process the query through the RAG service
        response = await rag_service.process_query_with_context(query_obj)
        
        # Format the response according to the API contract
        result = {
            "response_id": str(response.id),
            "content": response.content,
            "conversation_id": conversation_id or str(conversation_uuid),
            "timestamp": response.timestamp
        }
        
        # Include citations if requested
        if include_citations:
            result["citations"] = []
            # In a real implementation, we would properly format the citations
            # For now, we'll use a placeholder
            if hasattr(response, 'citations') and response.citations:
                for citation in response.citations:
                    if isinstance(citation, dict):
                        result["citations"].append(citation)
                    else:
                        result["citations"].append({
                            "source_reference": getattr(citation, 'source_reference', 'Unknown'),
                            "content_snippet": getattr(citation, 'content_snippet', '')[:200] + "..." 
                                if len(getattr(citation, 'content_snippet', '')) > 200 
                                else getattr(citation, 'content_snippet', ''),
                            "document_id": str(getattr(citation, 'document_id', 'unknown'))
                        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/conversation/start")
async def start_conversation(user_id: Optional[str] = None):
    """
    Start a new conversation
    """
    try:
        user_uuid = None
        if user_id:
            try:
                user_uuid = UUID(user_id)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid user ID format")
        
        conversation = await rag_service.start_conversation(user_uuid)
        
        return {
            "conversation_id": str(conversation.id),
            "timestamp": conversation.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting conversation: {str(e)}")


@router.get("/conversation/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """
    Get conversation history
    """
    try:
        # Validate conversation ID
        try:
            conv_uuid = UUID(conversation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid conversation ID format")
        
        # Get conversation history
        history = await rag_service.get_conversation_history(conv_uuid)
        
        return {
            "conversation_id": conversation_id,
            "messages": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving conversation: {str(e)}")


@router.post("/documents/index")
async def index_documents(
    source_path: str,
    title: Optional[str] = None,
    metadata: Optional[str] = None
):
    """
    Index book content for retrieval
    """
    try:
        # Parse metadata if provided
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON format")
        
        # Start the indexing process
        job_id = await rag_service.index_book_content(source_path, title, metadata_dict)
        
        return {
            "job_id": job_id,
            "status": "processing",
            "estimated_completion": "See status endpoint for updates"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting indexing: {str(e)}")


@router.get("/documents/index/status/{job_id}")
async def get_indexing_status(job_id: str):
    """
    Check indexing job status
    """
    try:
        # In a real implementation, this would check the actual job status
        # For now, we'll return a placeholder response
        return {
            "job_id": job_id,
            "status": "completed",  # Placeholder
            "progress": 1.0,  # Placeholder
            "details": "Indexing completed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking indexing status: {str(e)}")