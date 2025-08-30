---
title: career-chatbot-rag
app_file: app.py
sdk: gradio
sdk_version: 4.19.2
---

# 🚀 Career Chatbot with RAG

An intelligent career advisor chatbot powered by Retrieval-Augmented Generation (RAG) that provides personalized career guidance based on your professional profile and uploaded documents.

## ✨ Features

- **🤖 AI-Powered Career Advice**: Get personalized career guidance using OpenAI's GPT models
- **📄 Document Upload**: Upload PDF and text documents to enhance the chatbot's knowledge
- **🔍 Smart Context Retrieval**: Uses sentence embeddings to find relevant information from your documents
- **📚 Knowledge Base**: Automatically loads your LinkedIn profile and career summary
- **🎨 Modern UI**: Beautiful Gradio interface with responsive design
- **🔐 Authentication**: Simple password protection for privacy

## 🛠️ How It Works

1. **Document Loading**: Automatically loads your `linkedin.pdf` and `summary.txt` from the `me/` folder
2. **Text Processing**: Converts documents into searchable text chunks
3. **Embedding Generation**: Creates vector embeddings for semantic search
4. **Context Retrieval**: Finds most relevant information for each query
5. **AI Response**: Generates personalized career advice using retrieved context

## 📋 Required Files

The chatbot requires these files in the `me/` folder:
- `me/linkedin.pdf` - Your LinkedIn profile (PDF format)
- `me/summary.txt` - Your career summary (text format)

## 🚀 Deployment

This app is configured for Hugging Face Spaces deployment. Simply:

1. Upload your career documents to the `me/` folder
2. Set your `OPENAI_API_KEY` in the Space settings
3. The app will automatically deploy and be available

## 🔧 Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for AI capabilities

## 💡 Usage

1. **Start a conversation** by asking career-related questions
2. **Upload additional documents** to expand the knowledge base
3. **Get personalized advice** based on your profile and documents

## 🎯 Example Questions

- "What career path should I consider based on my background?"
- "How can I improve my LinkedIn profile?"
- "What skills should I develop for my target role?"
- "Can you analyze my career summary and suggest improvements?"

## 🔒 Security

- Simple password authentication (default: `career2024`)
- All processing happens locally in the Space
- No data is stored permanently

---

*Built with ❤️ using Gradio, OpenAI, and modern AI techniques*