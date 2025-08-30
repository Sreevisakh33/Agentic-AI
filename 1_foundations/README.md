---
title: career-chatbot-rag
app_file: app.py
sdk: gradio
sdk_version: 4.19.2
---

# Career Chatbot with RAG

A smart career chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about career, skills, and experience based on uploaded documents and LinkedIn profile.

## Features

- 🤖 AI-powered career chatbot
- 📚 RAG (Retrieval-Augmented Generation) for accurate responses
- 📄 Document upload and processing (PDF, TXT, DOCX, MD)
- 🔐 Password-protected document upload
- 💬 Interactive chat interface
- 🧠 Vector database with semantic search

## Required Files

**Important**: This app requires the following files to be present in the `me/` folder:
- `me/linkedin.pdf` - Your LinkedIn profile PDF
- `me/summary.txt` - Your career summary text

These files are processed on startup to build the initial knowledge base.

## How it Works

1. **Chat Interface**: Ask questions about career, skills, or experience
2. **RAG System**: The bot searches through uploaded documents to find relevant information
3. **Document Upload**: Add new documents to enhance the knowledge base (password protected)
4. **Smart Responses**: Get contextual answers based on the uploaded content

## Usage

1. Start chatting by asking questions about career, skills, or experience
2. The bot will search through the knowledge base and provide relevant answers
3. To upload documents, use the password: `RemembertheN@m3`

## Technical Details

- Built with Gradio for the web interface
- Uses sentence-transformers for text embeddings
- Implements cosine similarity for document retrieval
- OpenAI GPT-4 integration for intelligent responses
- In-memory vector database for fast search

## Local Development

```bash
pip install -r requirements.txt
python app.py
```

The app will be available at `http://127.0.0.1:7860`

## Deployment

This app is deployed on Hugging Face Spaces and can be accessed online.

**Note**: Ensure the `me/` folder with your career documents is included in the deployment.
