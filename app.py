import os
import gradio as gr
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import openai

class Me:
    def __init__(self):
        self.name = "Sreevisakh"
        self.linkedin = ""
        self.summary = ""
        self._load_career_documents()
        
    def _load_career_documents(self):
        """Load career documents from me folder"""
        try:
            # Load LinkedIn PDF
            if os.path.exists("me/linkedin.pdf"):
                with open("me/linkedin.pdf", "rb") as file:
                    pdf_reader = PdfReader(file)
                    self.linkedin = ""
                    for page in pdf_reader.pages:
                        self.linkedin += page.extract_text()
            
            # Load summary text
            if os.path.exists("me/summary.txt"):
                with open("me/summary.txt", "r", encoding="utf-8") as file:
                    self.summary = file.read()
                    
        except Exception as e:
            print(f"Error loading career documents: {e}")
            raise FileNotFoundError("Required career documents not found. Please ensure me/linkedin.pdf and me/summary.txt exist.")

class CareerChatbot:
    def __init__(self):
        self.me = Me()
        self.conversation_history = []
        self.uploaded_documents = []
        self.document_chunks = []
        self.embeddings = []
        
        # Initialize OpenAI client
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            print("Warning: OPENAI_API_KEY not found. Running in demo mode.")
        
        # Initialize sentence transformer
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.embedding_model = None
        
        # Build initial knowledge base
        self._build_knowledge_base()
    
    def _build_knowledge_base(self):
        """Build knowledge base from career documents"""
        documents = []
        
        # Add LinkedIn content
        if self.me.linkedin:
            documents.append(self.me.linkedin)
        
        # Add summary content
        if self.me.summary:
            documents.append(self.me.summary)
        
        # Add uploaded documents
        for doc in self.uploaded_documents:
            documents.append(doc)
        
        # Create chunks and embeddings
        self._create_chunks_and_embeddings(documents)
    
    def _create_chunks_and_embeddings(self, documents: List[str]):
        """Create text chunks and generate embeddings"""
        self.document_chunks = []
        self.embeddings = []
        
        for doc in documents:
            # Simple chunking by sentences (you can improve this)
            sentences = doc.split('. ')
            for sentence in sentences:
                if len(sentence.strip()) > 20:  # Minimum length
                    self.document_chunks.append(sentence.strip())
        
        # Generate embeddings
        if self.embedding_model and self.document_chunks:
            try:
                self.embeddings = self.embedding_model.encode(self.document_chunks)
            except Exception as e:
                print(f"Error generating embeddings: {e}")
    
    def _find_relevant_context(self, query: str, top_k: int = 3) -> str:
        """Find most relevant context for a query"""
        if not self.embedding_model or not self.embeddings:
            return ""
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k most similar chunks
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            relevant_context = []
            for idx in top_indices:
                if similarities[idx] > 0.3:  # Similarity threshold
                    relevant_context.append(self.document_chunks[idx])
            
            return " ".join(relevant_context)
        except Exception as e:
            print(f"Error finding relevant context: {e}")
            return ""
    
    def chat(self, message: str, history: List[List[str]]) -> tuple:
        """Handle chat messages"""
        if not message.strip():
            return "", history
        
        # Add user message to history
        history.append([message, ""])
        
        # Find relevant context
        relevant_context = self._find_relevant_context(message)
        
        # Prepare system message
        system_message = f"""You are a career advisor chatbot for {self.me.name}. 
        
        Career Information:
        LinkedIn: {self.me.linkedin[:500]}...
        Summary: {self.me.summary[:500]}...
        
        Relevant Context: {relevant_context}
        
        Provide helpful career advice based on the available information. Be professional, encouraging, and specific."""
        
        # Generate response
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": message}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                bot_response = response.choices[0].message.content
            except Exception as e:
                bot_response = f"I'm having trouble connecting to my AI service. Error: {str(e)}"
        else:
            bot_response = "I'm currently running in demo mode. Please set your OPENAI_API_KEY environment variable in Hugging Face Space settings to enable full AI capabilities."
        
        # Update history
        history[-1][1] = bot_response
        
        return "", history
    
    def process_uploaded_documents(self, files):
        """Process uploaded documents"""
        if not files:
            return "No files uploaded."
        
        processed_files = []
        for file in files:
            try:
                if file.name.endswith('.pdf'):
                    # Handle PDF files
                    pdf_reader = PdfReader(file.name)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    processed_files.append(text)
                elif file.name.endswith('.txt'):
                    # Handle text files
                    with open(file.name, 'r', encoding='utf-8') as f:
                        text = f.read()
                    processed_files.append(text)
                else:
                    processed_files.append(f"Unsupported file type: {file.name}")
            except Exception as e:
                processed_files.append(f"Error processing {file.name}: {str(e)}")
        
        self.uploaded_documents = processed_files
        self._build_knowledge_base()
        
        return f"Processed {len(processed_files)} document(s). Knowledge base updated!"

def create_simple_auth_interface():
    """Create a simple authentication interface"""
    def check_password(password):
        # Simple password check - you can make this more secure
        return password == "career2024"
    
    with gr.Blocks(title="Career Chatbot - Authentication") as auth_interface:
        gr.Markdown("# 🔐 Career Chatbot Authentication")
        gr.Markdown("Please enter the password to access the career chatbot.")
        
        password_input = gr.Textbox(label="Password", type="password")
        auth_button = gr.Button("Login", variant="primary")
        auth_output = gr.Textbox(label="Status", interactive=False)
        
        def authenticate(pwd):
            if check_password(pwd):
                return "✅ Authentication successful! Please close this tab and return to the main interface."
            else:
                return "❌ Invalid password. Please try again."
        
        auth_button.click(authenticate, inputs=[password_input], outputs=[auth_output])
    
    return auth_interface

def create_main_interface():
    """Create the main chatbot interface"""
    chatbot = CareerChatbot()
    
    with gr.Blocks(title="Career Chatbot with RAG", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🚀 Career Chatbot with RAG")
        gr.Markdown(f"**Welcome! I'm your AI career advisor for {chatbot.me.name}.**")
        
        # API Key warning
        if not os.getenv("OPENAI_API_KEY"):
            gr.Markdown("⚠️ **Warning**: OPENAI_API_KEY not found. Running in demo mode. Set this in Hugging Face Space settings for full AI capabilities.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot_interface = gr.Chatbot(height=500)
                msg_input = gr.Textbox(label="Ask me about your career", placeholder="What career advice do you need?")
                send_button = gr.Button("Send", variant="primary")
                
                # Clear button
                clear_button = gr.Button("Clear Chat")
            
            with gr.Column(scale=1):
                # Document upload
                gr.Markdown("## 📄 Upload Documents")
                gr.Markdown("Upload additional career-related documents to enhance my knowledge.")
                
                file_upload = gr.File(
                    label="Upload Documents",
                    file_types=[".pdf", ".txt"],
                    file_count="multiple"
                )
                upload_button = gr.Button("Process Documents")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
                
                # Knowledge base info
                gr.Markdown("## 📚 Knowledge Base")
                gr.Markdown(f"**LinkedIn Profile**: {'✅ Loaded' if chatbot.me.linkedin else '❌ Not found'}")
                gr.Markdown(f"**Summary**: {'✅ Loaded' if chatbot.me.summary else '❌ Not found'}")
                gr.Markdown(f"**Uploaded Documents**: {len(chatbot.uploaded_documents)}")
                gr.Markdown(f"**Total Chunks**: {len(chatbot.document_chunks)}")
        
        # Event handlers
        def send_message(message, history):
            return chatbot.chat(message, history)
        
        def clear_chat():
            return []
        
        def process_documents(files):
            return chatbot.process_uploaded_documents(files)
        
        # Connect events
        send_button.click(send_message, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface])
        msg_input.submit(send_message, inputs=[msg_input, chatbot_interface], outputs=[msg_input, chatbot_interface])
        clear_button.click(clear_chat, outputs=[chatbot_interface])
        upload_button.click(process_documents, inputs=[file_upload], outputs=[upload_status])
    
    return interface

# Create the main interface
iface = create_main_interface()

# Launch the app
if __name__ == "__main__":
    iface.launch(share=False)
