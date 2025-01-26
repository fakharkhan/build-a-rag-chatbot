# WonderVector5000 Chatbot

This application is a Retrieval-Augmented Generation (RAG) chatbot that combines the power of OpenAI's GPT-4 with Pinecone's vector database to provide context-aware responses about the WonderVector5000 system.

## Features

- **Context-Aware Responses**: Uses Pinecone vector store to retrieve relevant information before generating responses
- **LangChain Integration**: Implements LangChain's retrieval chain for efficient document handling
- **HuggingFace Embeddings**: Utilizes sentence-transformers for document embeddings
- **LangSmith Monitoring**: Integrated with LangSmith for performance monitoring and tracing

## Architecture Overview

1. **Pinecone Vector Store**: Stores and retrieves embeddings of the WonderVector5000 documentation
2. **HuggingFace Embeddings**: Converts text into vector representations for similarity search
3. **OpenAI GPT-4**: Generates natural language responses based on retrieved context
4. **LangChain**: Manages the retrieval and generation pipeline

## Setup Instructions

### Prerequisites

1. Python 3.8+
2. UV package manager
3. Required API keys (see Environment Variables section)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/wondervector5000-chatbot.git
   cd wondervector5000-chatbot
   ```

2. Install dependencies using UV:
   ```bash
   uv pip install -r requirements.txt
   ```

3. Set up environment variables (see next section)

### Environment Variables

Create a `.env` file with the following variables:

```env
PINECONE_API_KEY=your_pinecone_api_key
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
```

### Running the Application

Execute the main script:

```bash
python Chatbot.py
```

## Code Structure

- `Chatbot.py`: Main application logic
  - `setup_pinecone()`: Initializes Pinecone vector store
  - `Chatbot` class: Handles chat interactions
  - Main execution: Demonstrates both context-free and context-aware responses

## Example Usage

The application automatically runs a sample query comparing:
1. Direct GPT-4 response
2. Context-augmented response using Pinecone knowledge base

Sample output: 