from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]

class Me:
    def __init__(self):
        # Check if OpenAI API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("⚠️ Warning: OPENAI_API_KEY not found. Please set it in Hugging Face Space settings.")
            print("   The chatbot will work in demo mode without OpenAI integration.")
            self.openai = None
            self.demo_mode = True
        else:
            self.openai = OpenAI()
            self.demo_mode = False
            
        self.name = "Sreevisakh"
        
        # Try to read from me folder files, with fallback to sample data
        self.summary, self.linkedin = self._load_career_documents()
        
        # Initialize free RAG system with Hugging Face embeddings
        self._initialize_vector_database()
        
        # Initialize sentence transformer for embeddings
        print("🔄 Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Add existing documents to vector database
        self._initialize_documents()

    def _load_career_documents(self):
        """Load career documents from me folder"""
        try:
            # Get directory where this script is located
            base_path = os.path.dirname(os.path.abspath(__file__))
            
            # Build paths to the files
            pdf_path = os.path.join(base_path, "me", "linkedin.pdf")
            summary_path = os.path.join(base_path, "me", "summary.txt")
            
            # Read LinkedIn PDF
            linkedin_text = ""
            if os.path.exists(pdf_path):
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        linkedin_text += text
                print(f"✅ Successfully loaded LinkedIn PDF: {len(linkedin_text)} characters")
            else:
                raise FileNotFoundError(f"LinkedIn PDF not found at: {pdf_path}")
            
            # Read summary text
            summary_text = ""
            if os.path.exists(summary_path):
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary_text = f.read()
                print(f"✅ Successfully loaded summary: {len(summary_text)} characters")
            else:
                raise FileNotFoundError(f"Summary file not found at: {summary_path}")
            
            return summary_text, linkedin_text
            
        except Exception as e:
            print(f"❌ Critical Error: Could not load career documents: {e}")
            print("   Please ensure the 'me' folder contains 'linkedin.pdf' and 'summary.txt'")
            raise e

    def _initialize_vector_database(self):
        """Initialize in-memory vector database for RAG"""
        self.documents = []
        self.embeddings = []
        self.metadata = []
        print("✅ In-memory vector database initialized")

    def _initialize_documents(self):
        """Initialize vector database with existing documents"""
        if len(self.documents) > 0:
            print("✅ Documents already loaded in vector database")
            return
        
        print("📚 Processing existing documents...")
        
        # Process LinkedIn and summary documents
        documents = self._process_documents()
        
        # Add to vector database
        self._add_documents_to_vector_db(documents)
        print("✅ Document initialization complete")

    def _process_documents(self):
        """Process LinkedIn and summary documents into chunks"""
        documents = []
        
        # Process LinkedIn content
        linkedin_chunks = self._chunk_text(self.linkedin, chunk_size=600, overlap=100)
        for i, chunk in enumerate(linkedin_chunks):
            documents.append({
                'content': chunk,
                'source': 'linkedin',
                'chunk': i + 1,
                'type': 'profile'
            })
        
        # Process summary
        summary_chunks = self._chunk_text(self.summary, chunk_size=600, overlap=100)
        for i, chunk in enumerate(summary_chunks):
            documents.append({
                'content': chunk,
                'source': 'summary',
                'chunk': i + 1,
                'type': 'summary'
            })
        
        return documents

    def _chunk_text(self, text, chunk_size=600, overlap=100):
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks

    def _add_documents_to_vector_db(self, documents):
        """Add documents to in-memory vector database"""
        if not documents:
            return
        
        try:
            print(f"🔄 Generating embeddings for {len(documents)} documents...")
            
            for doc in documents:
                # Generate embedding
                embedding = self.embedding_model.encode(doc['content'])
                
                # Store document, embedding, and metadata
                self.documents.append(doc['content'])
                self.embeddings.append(embedding)
                self.metadata.append({
                    'source': doc['source'],
                    'type': doc['type'],
                    'chunk': doc['chunk']
                })
            
            print(f"✅ Added {len(documents)} documents to vector database")
            
        except Exception as e:
            print(f"❌ Error adding documents to vector database: {e}")

    def retrieve_relevant_context(self, query, n_results=3):
        """Retrieve relevant context using RAG with cosine similarity"""
        try:
            print(f"🔍 RAG Search: '{query}' - Retrieving {n_results} results")
            
            if len(self.documents) == 0:
                print("⚠️ No documents in vector database")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).reshape(1, -1)
            
            # Convert embeddings to numpy array for similarity calculation
            embeddings_array = np.array(self.embeddings)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, embeddings_array)[0]
            
            # Get top n_results most similar documents
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            contexts = []
            print(f" Found {len(top_indices)} relevant contexts")
            
            for i, idx in enumerate(top_indices):
                similarity_score = similarities[idx]
                content = self.documents[idx]
                meta = self.metadata[idx]
                
                print(f"  📄 Context {i+1}: {meta['source']} (Chunk {meta['chunk']}) - Score: {similarity_score:.3f}")
                print(f"      Preview: {content[:100]}...")
                
                contexts.append({
                    'source': meta['source'],
                    'content': content,
                    'metadata': meta,
                    'score': similarity_score
                })
            
            return contexts
            
        except Exception as e:
            print(f"❌ Error in RAG retrieval: {e}")
            return []

    def format_context_for_prompt(self, contexts):
        """Format retrieved contexts for inclusion in the prompt"""
        if not contexts:
            return "\n\n**No additional context available.**"
        
        context_text = "\n\n**📚 RELEVANT CONTEXT FROM KNOWLEDGE BASE:**\n"
        for i, context in enumerate(contexts, 1):
            source = context['metadata'].get('source', 'unknown')
            chunk = context['metadata'].get('chunk', 'unknown')
            score = context.get('score', 0)
            context_text += f"\n--- Context {i} (Source: {source}, Chunk {chunk}, Relevance: {score:.3f}) ---\n"
            context_text += f"{context['content']}\n"
        
        context_text += "\n**Use this context to provide accurate and detailed answers.**"
        return context_text

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. \
When the conversation starts, Greet the user saying Hello, I'm Digital Twin of {self.name} and can answer questions about {self.name}'s career, background, skills and experience."

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        """Enhanced chat function with RAG"""
        if not message.strip():
            return history, ""
        
        print(f"💬 Chat request: '{message}'")
        
        # Check if we're in demo mode
        if self.demo_mode:
            # Demo mode: provide RAG-based responses without OpenAI
            relevant_contexts = self.retrieve_relevant_context(message, n_results=3)
            if relevant_contexts:
                # Use the most relevant context to generate a simple response
                best_context = relevant_contexts[0]
                demo_response = f"Based on my knowledge base, I can tell you about {best_context['content'][:100]}...\n\n"
                demo_response += "Note: This is demo mode. To get full AI-powered responses, please set your OPENAI_API_KEY in the Hugging Face Space settings."
                return history + [[message, demo_response]], ""
            else:
                demo_response = "I'm sorry, I couldn't find relevant information in my knowledge base for that question.\n\n"
                demo_response += "Note: This is demo mode. To get full AI-powered responses, please set your OPENAI_API_KEY in the Hugging Face Space settings."
                return history + [[message, demo_response]], ""
        
        # Normal mode: use OpenAI with RAG
        relevant_contexts = self.retrieve_relevant_context(message, n_results=3)
        context_text = self.format_context_for_prompt(relevant_contexts)
        
        print(f"📚 RAG Context retrieved: {len(relevant_contexts)} contexts")
        
        # Create enhanced system prompt with RAG context
        enhanced_system_prompt = self.system_prompt() + context_text
        
        # Convert history to OpenAI format
        openai_history = []
        for user_msg, bot_msg in history:
            openai_history.append({"role": "user", "content": user_msg})
            openai_history.append({"role": "assistant", "content": bot_msg})
        
        messages = [{"role": "system", "content": enhanced_system_prompt}] + openai_history + [{"role": "user", "content": message}]
        
        print(f"🤖 Sending to OpenAI with {len(messages)} messages")
        
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            finish_reason = response.choices[0].finish_reason
            
            if finish_reason=="tool_calls":
                message_obj = response.choices[0].message
                tool_calls = message_obj.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message_obj)
                messages.extend(results)
            else:
                done = True
        
        bot_response = response.choices[0].message.content
        print(f" Bot response: {bot_response[:100]}...")
        
        return history + [[message, bot_response]], ""

    def process_uploaded_documents(self, files, auth_state):
        """Process uploaded documents and add to vector database"""
        print(f"📤 Upload attempt - Auth state: {auth_state}")
        
        if not auth_state:
            print("❌ Access denied - not authenticated")
            return "❌ Access denied! Please enter the correct password first."
        
        if not files:
            return "❌ No files selected for upload."
        
        print(f"📤 Processing {len(files)} files...")
        
        processed_count = 0
        errors = []
        
        for file in files:
            try:
                file_path = file.name
                file_extension = file_path.split('.')[-1].lower()
                print(f"📄 Processing: {file_path} (type: {file_extension})")
                
                if file_extension == 'pdf':
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    print(f"📊 Extracted {len(text)} characters from PDF")
                elif file_extension == 'txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    print(f"📝 Read {len(text)} characters from TXT")
                elif file_extension in ['docx', 'md']:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    print(f"📄 Read {len(text)} characters from {file_extension}")
                else:
                    errors.append(f"Unsupported file type: {file_extension}")
                    continue
                
                chunks = self._chunk_text(text, chunk_size=800, overlap=150)
                print(f"✂️ Created {len(chunks)} chunks")
                
                # Add to vector database
                documents = []
                for i, chunk in enumerate(chunks):
                    documents.append({
                        'content': chunk,
                        'source': 'uploaded',
                        'type': 'document',
                        'filename': file_path.split('/')[-1],
                        'chunk': i + 1,
                        'uploaded': True
                    })
                
                self._add_documents_to_vector_db(documents)
                processed_count += 1
                print(f"✅ Successfully added {file_path} to database")
                
            except Exception as e:
                error_msg = f"Error processing {file.name}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ {error_msg}")
        
        status_msg = f"✅ Successfully processed {processed_count} document(s)"
        if errors:
            status_msg += f"\n❌ Errors: {len(errors)}"
            for error in errors[:3]:
                status_msg += f"\n  - {error}"
        
        print(f"📊 Final status: {status_msg}")
        return status_msg

    def create_simple_auth_interface(self):
        """Simple chat interface with basic password authentication and document upload"""
        with gr.Blocks(title="Sreevisakh's Career Chatbot") as demo:
            gr.Markdown("#Sreevisakh's Career Chatbot")
            gr.Markdown("Chat about careers and upload documents to enhance knowledge base")

            # Authentication state
            auth_state = gr.State(False)
            
            with gr.Row():
                # Main Chat Interface (Always visible)
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(height=500)
                    msg = gr.Textbox(label="Ask me about my career, skills, or experience")
                    
                    # Button row with Send and Clear
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear Chat", variant="secondary")
                
                # Right Side Panel - Document Upload
                with gr.Column(scale=1):
                    gr.Markdown("## Document Upload to enhance knowledge base")
                    gr.Markdown("**Restricted Access - Password Required**")
                    
                    # Simple Password Authentication
                    password_input = gr.Textbox(label="Password", placeholder="Enter password", type="password")
                    auth_status_message = gr.Textbox(label="Status", interactive=False)
                    verify_password_btn = gr.Button("Verify Password", variant="primary")
                    
                    # Upload interface (initially hidden)
                    upload_interface = gr.Group(visible=False)
                    with upload_interface:
                        gr.Markdown("🎉 **Access Granted!**")
                        gr.Markdown("**Upload your documents below:**")
                        file_upload = gr.File(
                            label="Choose Files to Upload", 
                            file_types=[".pdf", ".txt", ".docx", ".md"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("Process & Add to Knowledge Base", variant="primary")
                        upload_status = gr.Textbox(label="Upload Status", interactive=False)
                        
                        # Logout
                        logout_btn = gr.Button("Logout", variant="stop")
            
            # Simple Authentication Functions
            def verify_password(password):
                """Simple password verification and UI update logic"""
                UPLOAD_PASSWORD = "RemembertheN@m3"
                if password == UPLOAD_PASSWORD:
                    print(f"✅ Password authentication successful")
                    return True, "🎉 Access granted! You can now upload documents.", gr.update(visible=True)
                else:
                    print(f"❌ Password authentication failed")
                    return False, "❌ Invalid password. Please try again.", gr.update(visible=False)

            # UI Event Handlers
            def on_logout():
                """Hide upload interface and reset authentication"""
                print("🔓 Logging out - hiding upload interface")
                return gr.update(visible=False), False

            # Connect events
            verify_password_btn.click(
                verify_password,
                inputs=[password_input],
                outputs=[auth_state, auth_status_message, upload_interface]
            )
            
            # Process uploads only if password authenticated
            upload_btn.click(
                self.process_uploaded_documents,
                inputs=[file_upload, auth_state],
                outputs=[upload_status]
            )
            
            # Chat events - both Enter key and Send button work, with message clearing
            msg.submit(self.chat, [msg, chatbot], [chatbot, msg])
            send_btn.click(self.chat, [msg, chatbot], [chatbot, msg])
            
            clear.click(lambda: None, None, chatbot, queue=False)
            
            # Logout resets both interface and authentication state
            logout_btn.click(
                on_logout,
                outputs=[upload_interface, auth_state]
            )
        
        return demo

if __name__ == "__main__":
    me = Me()
    # Launch the enhanced interface with authentication and document upload
    demo = me.create_simple_auth_interface()
    demo.launch()
    