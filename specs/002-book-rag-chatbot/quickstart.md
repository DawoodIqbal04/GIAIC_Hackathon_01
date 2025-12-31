# Quickstart Guide: Book RAG Chatbot

## Prerequisites

- Python 3.11+
- Node.js 16+ (for Docusaurus)
- Access to OpenAI API
- Access to Qdrant Cloud (Free Tier)
- Access to Neon Serverless Postgres

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys and database connection strings
```

### 3. Frontend Setup

```bash
# Navigate to the website directory
cd website

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your API endpoints and frontend settings
```

## Configuration

### Backend Configuration

1. Update `backend/.env` with:
   - OpenAI API key
   - Qdrant Cloud endpoint and API key
   - Neon Serverless Postgres connection string
   - Other application settings

2. Initialize the database:
```bash
python -m src.database.init
```

### Frontend Configuration

1. Update `website/.env` with:
   - Backend API endpoint
   - Frontend-specific settings

## Running the Application

### Backend (API Server)

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m src.api.main
```

The backend server will start on `http://localhost:8000`

### Frontend (Docusaurus)

```bash
cd website
npm start
```

The website will start on `http://localhost:3000`

## Indexing Book Content

To index your book content for RAG:

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m src.scripts.index_book --source-path /path/to/book/content
```

This will process the book content and store vector embeddings in Qdrant.

## Using the Chatbot

1. Access the Docusaurus website at `http://localhost:3000`
2. Navigate to the chatbot interface
3. Type your questions about the book content
4. The chatbot will retrieve relevant information and generate responses
5. Citations will be provided for the information sources

## Selected-Text Mode

To use selected-text mode:

1. Highlight text in the book content
2. Activate the selected-text mode in the chatbot interface
3. Ask questions specifically about the selected text
4. The chatbot will restrict its answers to the selected text only

## Testing

### Backend Tests

```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
pytest
```

### Frontend Tests

```bash
cd website
npm test
```

## Troubleshooting

### Common Issues

- **API Connection Errors**: Verify backend server is running and API endpoint is correctly configured in frontend
- **Embedding Issues**: Ensure Qdrant Cloud is properly configured and book content is indexed
- **Database Connection Errors**: Check Neon Serverless Postgres connection settings in backend configuration