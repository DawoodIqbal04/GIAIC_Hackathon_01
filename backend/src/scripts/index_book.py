"""
Script to index book content for the RAG chatbot
"""
import argparse
import asyncio
import os
from pathlib import Path
from typing import List
from uuid import UUID

from ..models.document import Document
from ..services.retrieval_service import RetrievalService
from ..utils.helpers import generate_uuid


async def index_book_content(source_path: str, title: str = None, metadata: dict = None):
    """
    Index book content from a given source path
    """
    print(f"Starting to index book content from: {source_path}")
    
    # Initialize the retrieval service
    retrieval_service = RetrievalService()
    
    # Determine the title if not provided
    if not title:
        title = Path(source_path).stem
    
    # Default metadata
    if not metadata:
        metadata = {}
    
    # Check if source_path is a directory or file
    source = Path(source_path)
    if source.is_dir():
        # Process all markdown files in the directory
        files = list(source.glob("**/*.md")) + list(source.glob("**/*.mdx"))
    elif source.is_file() and source.suffix in [".md", ".mdx"]:
        files = [source]
    else:
        print(f"Error: {source_path} is not a valid markdown file or directory")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process each file
    for file_path in files:
        print(f"Processing: {file_path}")
        
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Chunk the content into smaller pieces (max 10k characters per chunk)
        chunks = chunk_content(content)
        
        # Create and index Document objects for each chunk
        for i, chunk in enumerate(chunks):
            doc_title = f"{title} - {file_path.name} - Chunk {i+1}"
            document = Document(
                title=doc_title,
                content=chunk,
                source_path=str(file_path),
                metadata={
                    **metadata,
                    "original_file": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            
            # Index the document
            success = await retrieval_service.index_document(document)
            if success:
                print(f"  Indexed chunk {i+1}/{len(chunks)}")
            else:
                print(f"  Failed to index chunk {i+1}")
    
    print("Indexing completed!")


def chunk_content(content: str, max_chunk_size: int = 8000) -> List[str]:
    """
    Split content into chunks of maximum size
    """
    chunks = []
    paragraphs = content.split('\n\n')
    
    current_chunk = ""
    for paragraph in paragraphs:
        # If adding this paragraph would exceed the limit
        if len(current_chunk) + len(paragraph) > max_chunk_size:
            # If the current chunk isn't empty, save it
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If the paragraph itself is larger than the limit, split it
            if len(paragraph) > max_chunk_size:
                sub_chunks = split_large_paragraph(paragraph, max_chunk_size)
                chunks.extend(sub_chunks[:-1])  # Add all but the last sub-chunk
                current_chunk = sub_chunks[-1]  # Keep the last sub-chunk
            else:
                current_chunk = paragraph
        else:
            current_chunk += f"\n\n{paragraph}"
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def split_large_paragraph(paragraph: str, max_chunk_size: int) -> List[str]:
    """
    Split a large paragraph into smaller chunks
    """
    sentences = paragraph.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence_with_period = sentence + '. '
        
        if len(current_chunk) + len(sentence_with_period) > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence_with_period
        else:
            current_chunk += sentence_with_period
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Index book content for the RAG chatbot")
    parser.add_argument(
        "--source-path", 
        required=True, 
        help="Path to the book content (directory or file)"
    )
    parser.add_argument(
        "--title", 
        help="Title of the book (optional, will be inferred from path if not provided)"
    )
    parser.add_argument(
        "--metadata", 
        help="Additional metadata as JSON string (optional)"
    )
    
    args = parser.parse_args()
    
    # Parse metadata if provided
    metadata = None
    if args.metadata:
        import json
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            print("Error: Invalid JSON in metadata")
            return
    
    # Run the indexing process
    asyncio.run(index_book_content(args.source_path, args.title, metadata))


if __name__ == "__main__":
    main()