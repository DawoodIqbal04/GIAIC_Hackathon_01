# Vector Database Setup for Content Indexing

## Overview
This document outlines the setup process for the vector database to support content indexing in the RAG system for the Physical AI and Humanoid Robotics Technical Book.

## Vector Database Options

### Option 1: Pinecone (Cloud-based)
- Managed service with high availability
- Automatic scaling
- Pay-per-use pricing model
- Good for production deployments

### Option 2: Weaviate (Open-source, Self-hosted)
- Open-source with commercial support available
- GraphQL and REST APIs
- Can be self-hosted or cloud-deployed
- Good for custom deployments

### Option 3: Chroma (Local/Lightweight)
- Lightweight, embeddable vector database
- Good for development and testing
- Python-native implementation
- Can scale to production with proper configuration

## Recommended Setup: Weaviate

For this implementation, we'll focus on Weaviate as it provides a good balance of features, open-source licensing, and deployment flexibility.

### Docker Compose Setup

Create a `docker-compose.yml` file for Weaviate:

```yaml
version: '3.4'
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: semitechnologies/weaviate:1.22.5
    ports:
    - 8080:8080
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformers:8080'
      ENABLE_MODULES: 'text2vec-transformers'
      CLUSTER_HOSTNAME: 'node1'
    depends_on:
      - t2v-transformers
  t2v-transformers:
    image: semitechnologies/transformers-inference:sentence-transformers-all-MiniLM-L6-v2
    environment:
      ENABLE_CUDA: '0'  # Set to 1 if using GPU
    ports:
      - 8081:8080
```

### Schema Definition

Define the schema for storing document chunks:

```json
{
  "class": "DocChunk",
  "description": "A chunk of documentation content",
  "properties": [
    {
      "name": "content",
      "description": "The text content of the chunk",
      "dataType": ["text"]
    },
    {
      "name": "sourceDocument",
      "description": "The original document this chunk comes from",
      "dataType": ["string"]
    },
    {
      "name": "sectionTitle",
      "description": "The section title of this chunk",
      "dataType": ["string"]
    },
    {
      "name": "module",
      "description": "The module this chunk belongs to",
      "dataType": ["string"]
    },
    {
      "name": "chapter",
      "description": "The chapter this chunk belongs to",
      "dataType": ["string"]
    },
    {
      "name": "chunkIndex",
      "description": "The position of this chunk in the document",
      "dataType": ["int"]
    }
  ]
}
```

## Implementation Steps

### 1. Install Weaviate Client
```bash
pip install weaviate-client
```

### 2. Initialize Weaviate Client
```python
import weaviate
import os

# Connect to Weaviate instance
client = weaviate.Client(
    url = "http://localhost:8080",  # Replace with your Weaviate URL
    # Additional parameters if authentication is enabled
)
```

### 3. Create the Schema
```python
# Define the class
class_obj = {
    "class": "DocChunk",
    "description": "A chunk of documentation content",
    "properties": [
        {
            "name": "content",
            "description": "The text content of the chunk",
            "dataType": ["text"]
        },
        {
            "name": "sourceDocument",
            "description": "The original document this chunk comes from",
            "dataType": ["string"]
        },
        {
            "name": "sectionTitle",
            "description": "The section title of this chunk",
            "dataType": ["string"]
        },
        {
            "name": "module",
            "description": "The module this chunk belongs to",
            "dataType": ["string"]
        },
        {
            "name": "chapter",
            "description": "The chapter this chunk belongs to",
            "dataType": ["string"]
        },
        {
            "name": "chunkIndex",
            "description": "The position of this chunk in the document",
            "dataType": ["int"]
        }
    ]
}

# Create the class
client.schema.create_class(class_obj)
```

### 4. Indexing Content
```python
def index_document_chunk(content, source_doc, section_title, module, chapter, chunk_index):
    """
    Index a document chunk in Weaviate
    """
    data_obj = {
        "content": content,
        "sourceDocument": source_doc,
        "sectionTitle": section_title,
        "module": module,
        "chapter": chapter,
        "chunkIndex": chunk_index
    }
    
    client.data_object.create(
        data_object=data_obj,
        class_name="DocChunk"
    )

# Example usage
index_document_chunk(
    content="This is a sample document chunk",
    source_doc="module-1-intro.md",
    section_title="Introduction to ROS 2",
    module="Module 1: ROS 2 and Robotic Control Foundations",
    chapter="Chapter 1: Middleware for robot control",
    chunk_index=0
)
```

### 5. Querying Content
```python
def search_content(query, limit=5):
    """
    Search for content in the vector database
    """
    result = (
        client.query
        .get("DocChunk", ["content", "sourceDocument", "sectionTitle", "module", "chapter"])
        .with_near_text({"concepts": [query]})
        .with_limit(limit)
        .do()
    )
    
    return result["data"]["Get"]["DocChunk"]

# Example usage
results = search_content("What is ROS 2 middleware?", limit=3)
for result in results:
    print(f"Module: {result['module']}")
    print(f"Chapter: {result['chapter']}")
    print(f"Section: {result['sectionTitle']}")
    print(f"Content: {result['content'][:200]}...")
    print("---")
```

## Alternative: Using ChromaDB for Development

For development and testing, you might prefer a simpler setup with ChromaDB:

### Install ChromaDB
```bash
pip install chromadb
```

### ChromaDB Setup
```python
import chromadb
from chromadb.utils import embedding_functions

# Initialize Chroma client
client = chromadb.PersistentClient(path="./chroma_data")

# Create collection with sentence transformer embedding function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="doc_chunks",
    embedding_function=sentence_transformer_ef
)

# Add documents
collection.add(
    documents=["This is a sample document chunk"],
    metadatas=[{
        "source_document": "module-1-intro.md",
        "section_title": "Introduction to ROS 2",
        "module": "Module 1: ROS 2 and Robotic Control Foundations",
        "chapter": "Chapter 1: Middleware for robot control"
    }],
    ids=["chunk_0"]
)

# Query documents
results = collection.query(
    query_texts=["What is ROS 2 middleware?"],
    n_results=3
)

print(results)
```

## Production Considerations

1. **Scalability**: Ensure the vector database can handle expected query loads
2. **Backup**: Implement regular backups of vector database
3. **Monitoring**: Set up monitoring for performance and availability
4. **Security**: Implement proper authentication and authorization if needed
5. **Cost**: Monitor usage costs, especially for cloud-based solutions

## Testing the Setup

Create a simple test script to verify the vector database is working:

```python
import weaviate
import os

def test_vector_db():
    # Connect to Weaviate
    client = weaviate.Client(
        url="http://localhost:8080"
    )
    
    # Test connection
    try:
        client.schema.get("DocChunk")
        print("Schema exists, connection successful")
    except:
        print("Creating new schema...")
        # Create schema here if needed
    
    # Add a test document
    test_data = {
        "content": "This is a test document for the vector database",
        "sourceDocument": "test-doc.md",
        "sectionTitle": "Test Section",
        "module": "Test Module",
        "chapter": "Test Chapter",
        "chunkIndex": 0
    }
    
    client.data_object.create(
        data_object=test_data,
        class_name="DocChunk"
    )
    
    print("Test document added successfully")

if __name__ == "__main__":
    test_vector_db()
```

This setup provides a robust foundation for the RAG system's content indexing needs, supporting semantic search and retrieval of documentation content for the chatbot functionality.