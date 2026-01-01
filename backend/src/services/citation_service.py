"""
Citation service for the Book RAG Chatbot
Handles the creation and management of citations for generated responses
"""
from typing import List
from uuid import UUID
from ..models.citation import Citation
from ..models.retrieved_context import RetrievedContext


class CitationService:
    def __init__(self):
        pass

    def create_citations_from_contexts(
        self, 
        response_id: UUID, 
        retrieved_contexts: List[RetrievedContext]
    ) -> List[Citation]:
        """
        Create citation objects from retrieved contexts for a response
        """
        citations = []
        for ctx in retrieved_contexts:
            citation = Citation(
                response_id=response_id,
                document_id=ctx.document_id,
                source_reference=ctx.source_citation,
                content_snippet=ctx.content[:200] + "..." if len(ctx.content) > 200 else ctx.content
            )
            citations.append(citation)
        
        return citations

    def format_citations_for_response(
        self, 
        citations: List[Citation]
    ) -> List[dict]:
        """
        Format citations for inclusion in API response
        """
        formatted_citations = []
        for citation in citations:
            formatted_citations.append({
                "source_reference": citation.source_reference,
                "content_snippet": citation.content_snippet,
                "document_id": str(citation.document_id)
            })
        
        return formatted_citations

    def validate_citation_format(self, citation_data: dict) -> bool:
        """
        Validate that citation data has the required fields
        """
        required_fields = ["source_reference", "content_snippet", "document_id"]
        for field in required_fields:
            if field not in citation_data:
                return False
        return True

    def merge_duplicate_citations(self, citations: List[dict]) -> List[dict]:
        """
        Merge citations that refer to the same source
        """
        unique_citations = {}
        for citation in citations:
            source_ref = citation.get("source_reference", "")
            if source_ref in unique_citations:
                # If we already have a citation for this source, combine the snippets
                existing_snippet = unique_citations[source_ref]["content_snippet"]
                new_snippet = citation["content_snippet"]
                # Combine snippets, avoiding duplication
                if new_snippet not in existing_snippet:
                    unique_citations[source_ref]["content_snippet"] += f" ... {new_snippet}"
            else:
                unique_citations[source_ref] = citation
        
        return list(unique_citations.values())