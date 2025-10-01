# Generative-AI-Project - RAG Chatbot with OpenAI

A unified RAG (Retrieval-Augmented Generation) chatbot application supporting:
- Website content analysis with Groq
- PDF document chat with Groq  
- General conversation with OpenAI GPT models

## Quick Start

### Build Docker Image
```bash
docker build -t rag-chatbot .
```

### Run with OpenAI API Key
```bash
# Option 1: Pass API key as environment variable
docker run -p 8501:8501 -e OPENAI_API_KEY=your_openai_key_here rag-chatbot

# Option 2: Use .env file
docker run -p 8501:8501 --env-file .env rag-chatbot
```

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features
- 🌐 **Website RAG**: Chat with any website content
- 📄 **PDF RAG**: Upload and chat with PDF documents
- 🤖 **OpenAI Chat**: GPT-4o-mini powered conversations
- 🔑 **API Key Management**: Secure key input and validation
- 🐳 **Docker Ready**: Containerized for easy deployment
