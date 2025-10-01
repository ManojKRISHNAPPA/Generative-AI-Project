# Generative-AI-Project
docker build -t chatbot-app .

# Run with ports for Streamlit + Ollama
docker run -p 8501:8501 -p 11434:11434 --env-file .env chatbot-app