# ta7t-bety-RAG System-chatbot

A Retrieval-Augmented Generation (RAG) system built with Flask, LangChain, and Ollama LLM to provide intelligent responses based on various data sources including MongoDB collections and text files.

## Features

- **Intelligent Chatbot**: Answers questions by retrieving relevant information from multiple data sources
- **Nearby Restaurant Finder**: Locates restaurants within a specified radius using geospatial calculations
- **Data Refresh Capability**: Updates the knowledge base with the latest information
- **Health Check Monitoring**: Provides system status for monitoring

## Architecture

The system uses a combination of technologies:

- **Flask**: Web server for API endpoints
- **LangChain**: Framework for building LLM applications
- **Ollama**: Local LLM for text generation (using llama3.2)
- **FAISS**: Vector database for efficient similarity search
- **MongoDB**: Document database for storing structured data
- **PyMongo**: MongoDB client for Python

## Data Sources

The system retrieves information from:

1. MongoDB collections:
   - `users`: User profile information
   - `providers`: Service provider details including geolocation
   - `knowledge_base`: General knowledge information
   - `posts`: User-generated content

2. Local text file:
   - Chatbot document: Additional knowledge stored in a text file

## API Endpoints

### Chat API
```
POST /api/chat
```
Request body:
```json
{
  "question": "What restaurants are near me?"
}
```
Response:
```json
{
  "success": true,
  "question": "What restaurants are near me?",
  "answer": "To find restaurants near you, I would need your location coordinates..."
}
```

### Find Nearby Restaurants
```
POST /api/find_restaurants
```
Request body:
```json
{
  "latitude": 31.2001,
  "longitude": 29.9187,
  "radius": 5
}
```
Response:
```json
{
  "success": true,
  "message": "Found 3 restaurants near you",
  "restaurants": [
    {
      "name": "Restaurant A",
      "address": "123 Main St",
      "distance_km": 1.2,
      "coordinates": {
        "latitude": 31.2024,
        "longitude": 29.9254
      }
    },
    ...
  ]
}
```

### Refresh Data
```
POST /api/refresh
```
Response:
```json
{
  "success": true,
  "message": "Data updated successfully"
}
```

### Health Check
```
GET /api/health
```
Response:
```json
{
  "status": "ok",
  "message": "RAG system is operational",
  "components": {
    "mongodb": true,
    "vector_store": true,
    "rag_chain": true,
    "conversational_chain": true
  }
}
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- MongoDB Atlas account (or local MongoDB server)
- Ollama with llama3.2 model installed

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ta7t-bety.git
   cd ta7t-bety
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables or update connection string in the code:
   ```python
   # Example MongoDB connection string in the code
   client = pymongo.MongoClient("mongodb+srv**************************************", 
                            %%%%%%%%%%%%%%%%%%=5000)
   ```

4. Update the text file path:
   ```python
   TEXT_FILE_PATH = r"path/to/your/chatbot document.txt"
   ```

5. Run the server:
   ```
   python app.py
   ```

The server will start on port 5000 by default.

### Running with Ollama

Make sure Ollama is running and the required models are available:
- all-minilm (for embeddings)
- llama3.2 (for text generation)

Run the Ollama server before starting the application:
```
ollama serve
```

## System Design

The system initializes in the background to avoid blocking the application startup. It loads documents from MongoDB and text files, processes them into chunks, and builds a vector store using FAISS with embeddings from Ollama.

Two query chains are created:
1. **RAG Chain**: For answering specific questions by retrieving relevant context
2. **Conversational Chain**: For handling simple conversational queries

The system automatically determines which chain to use based on the nature of the query.

## Error Handling

The application includes comprehensive error handling:
- MongoDB connection failures
- Ollama service unavailability
- Vector store creation issues
- Request processing errors

## Project Structure

```
├── app.py             # Main application file
├── faiss_index/       # Directory for storing FAISS index
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
```

## Requirements

- flask
- langchain
- langchain-community
- langchain-ollama
- pymongo
- faiss-cpu
- geopy
- ollama
