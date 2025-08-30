# RAG Implementation for Enhanced Career Knowledge

This code can be copied into your `4_lab4.ipynb` notebook as new cells. It enhances your existing `chat` function with RAG capabilities while maintaining the same interface.

## 1. Enhanced Imports and Setup

```python
import chromadb
from chromadb.config import Settings
import numpy as np
from typing import List, Dict, Any
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
```

## 2. Document Processing and Chunking

```python
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks for better retrieval
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks

def process_documents():
    """
    Process and chunk the existing documents for RAG
    """
    # Process LinkedIn profile
    reader = PdfReader("me/linkedin.pdf")
    linkedin_chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            chunks = chunk_text(text, chunk_size=800, overlap=150)
            for j, chunk in enumerate(chunks):
                linkedin_chunks.append({
                    'content': chunk,
                    'source': 'linkedin',
                    'page': i + 1,
                    'chunk': j + 1,
                    'type': 'profile'
                })
    
    # Process summary
    with open("me/summary.txt", "r", encoding="utf-8") as f:
        summary_text = f.read()
    
    summary_chunks = chunk_text(summary_text, chunk_size=600, overlap=100)
    summary_docs = []
    for i, chunk in enumerate(summary_chunks):
        summary_docs.append({
            'content': chunk,
            'source': 'summary',
            'chunk': i + 1,
            'type': 'summary'
        })
    
    return linkedin_chunks + summary_docs

# Process documents
documents = process_documents()
print(f"Processed {len(documents)} document chunks")
```

## 3. ChromaDB Vector Database Setup

```python
# Initialize ChromaDB with persistent storage
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = client.get_or_create_collection(
    name="career_knowledge",
    metadata={"hnsw:space": "cosine"}
)

# Add documents to the vector database
def add_documents_to_chroma():
    """
    Add processed documents to ChromaDB
    """
    if len(collection.get()['ids']) > 0:
        print("Documents already in database, skipping...")
        return
    
    ids = []
    texts = []
    metadatas = []
    
    for i, doc in enumerate(documents):
        doc_id = f"doc_{i}_{doc['source']}_{doc['chunk']}"
        ids.append(doc_id)
        texts.append(doc['content'])
        metadatas.append({
            'source': doc['source'],
            'type': doc['type'],
            'chunk': doc['chunk'],
            'page': doc.get('page', 'N/A')
        })
    
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Added {len(ids)} documents to ChromaDB")

# Add documents
add_documents_to_chroma()
```

## 4. Enhanced Retrieval Functions

```python
def retrieve_relevant_context(query: str, n_results: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve relevant context from the vector database
    """
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        contexts = []
        for i in range(len(results['documents'][0])):
            contexts.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return contexts
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return []

def format_context_for_prompt(contexts: List[Dict[str, Any]]) -> str:
    """
    Format retrieved contexts for inclusion in the prompt
    """
    if not contexts:
        return ""
    
    formatted = "\n\n## Relevant Context:\n"
    for i, ctx in enumerate(contexts, 1):
        source = ctx['metadata']['source']
        chunk = ctx['metadata']['chunk']
        content = ctx['content'][:300] + "..." if len(ctx['content']) > 300 else ctx['content']
        
        formatted += f"\n{i}. Source: {source} (Chunk {chunk})\n"
        formatted += f"   Content: {content}\n"
    
    return formatted
```

## 5. Enhanced Chat Function with RAG (Modified to work with existing interface)

```python
# MODIFY YOUR EXISTING chat FUNCTION - Replace the current one with this enhanced version
def chat(message, history):
    """
    Enhanced chat function that uses RAG for better context
    Maintains the same interface as your original function
    """
    # Retrieve relevant context using RAG
    relevant_contexts = retrieve_relevant_context(message, n_results=2)
    context_text = format_context_for_prompt(relevant_contexts)
    
    # Create enhanced system prompt with RAG context
    enhanced_system_prompt = system_prompt + context_text
    
    messages = [{"role": "system", "content": enhanced_system_prompt}] + history + [{"role": "user", "content": message}]
    done = False
    while not done:

        # This is the call to the LLM - see that we pass in the tools json
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)

        finish_reason = response.choices[0].finish_reason
        
        # If the LLM wants to call a tool, we do that!
        if finish_reason=="tool_calls":
            message_obj = response.choices[0].message
            tool_calls = message_obj.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message_obj)
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content

# Test the enhanced RAG system
test_query = "What are your technical skills and experience?"
print("Testing RAG retrieval...")
contexts = retrieve_relevant_context(test_query)
print(f"Retrieved {len(contexts)} relevant contexts")
for i, ctx in enumerate(contexts):
    print(f"\nContext {i+1}:")
    print(f"Source: {ctx['metadata']['source']}")
    print(f"Content: {ctx['content'][:200]}...")
```

## 6. Advanced RAG Features

```python
def semantic_search_with_filters(query: str, source_filter: str = None, type_filter: str = None, n_results: int = 5):
    """
    Advanced semantic search with metadata filtering
    """
    where_clause = {}
    if source_filter:
        where_clause['source'] = source_filter
    if type_filter:
        where_clause['type'] = type_filter
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_clause if where_clause else None,
            include=['documents', 'metadatas', 'distances']
        )
        return results
    except Exception as e:
        print(f"Error in filtered search: {e}")
        return None

def add_custom_knowledge(content: str, source: str, knowledge_type: str):
    """
    Add custom knowledge to the RAG system
    """
    chunks = chunk_text(content, chunk_size=800, overlap=150)
    
    ids = []
    texts = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        doc_id = f"custom_{source}_{i}_{int(np.random.random() * 10000)}"
        ids.append(doc_id)
        texts.append(chunk)
        metadatas.append({
            'source': source,
            'type': knowledge_type,
            'chunk': i + 1,
            'custom': True
        })
    
    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"Added {len(ids)} custom knowledge chunks")

# Example: Add some custom knowledge
custom_knowledge = """
I am passionate about AI and machine learning, with experience in building conversational AI systems.
I enjoy working on projects that combine technical innovation with practical business applications.
My expertise includes Python, JavaScript, and cloud technologies.
"""
add_custom_knowledge(custom_knowledge, "personal_interests", "interests")
```

## 7. RAG Analytics and Monitoring

```python
def get_rag_statistics():
    """
    Get statistics about the RAG system
    """
    try:
        all_docs = collection.get()
        total_docs = len(all_docs['ids'])
        
        # Count by source
        sources = {}
        types = {}
        for metadata in all_docs['metadatas']:
            source = metadata['source']
            doc_type = metadata['type']
            
            sources[source] = sources.get(source, 0) + 1
            types[doc_type] = types.get(doc_type, 0) + 1
        
        print(f"Total documents: {total_docs}")
        print(f"Sources: {sources}")
        print(f"Types: {types}")
        
        return {
            'total_docs': total_docs,
            'sources': sources,
            'types': types
        }
    except Exception as e:
        print(f"Error getting statistics: {e}")
        return None

# Get RAG statistics
print("\nRAG System Statistics:")
rag_stats = get_rag_statistics()
```

## 8. Enhanced Gradio Interface with RAG (Optional - Enhanced version)

```python
def create_enhanced_interface():
    """
    Create an enhanced Gradio interface with RAG capabilities
    This is optional - you can keep using your existing interface
    """
    with gr.Blocks(title="Enhanced Career Chatbot with RAG") as demo:
        gr.Markdown("# Enhanced Career Chatbot with RAG")
        gr.Markdown("This chatbot uses RAG to provide more accurate and context-aware responses.")
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=600)
                msg = gr.Textbox(label="Ask me about my career, skills, or experience")
                clear = gr.Button("Clear")
                
            with gr.Column(scale=1):
                gr.Markdown("## RAG System Info")
                stats_btn = gr.Button("Show RAG Stats")
                stats_output = gr.Textbox(label="Statistics", interactive=False)
                
                gr.Markdown("## Add Custom Knowledge")
                custom_content = gr.Textbox(label="Custom knowledge content", lines=3)
                custom_source = gr.Textbox(label="Source name")
                custom_type = gr.Textbox(label="Knowledge type")
                add_knowledge_btn = gr.Button("Add Knowledge")
                
                gr.Markdown("## Search Knowledge Base")
                search_query = gr.Textbox(label="Search query")
                search_results = gr.Textbox(label="Search results", lines=5, interactive=False)
                search_btn = gr.Button("Search")
        
        def respond(message, history):
            return chat(message, history)  # Uses your enhanced chat function
        
        def show_stats():
            stats = get_rag_statistics()
            if stats:
                return f"Total docs: {stats['total_docs']}\nSources: {stats['sources']}\nTypes: {stats['types']}"
            return "Error retrieving statistics"
        
        def add_knowledge(content, source, ktype):
            if content and source and ktype:
                add_custom_knowledge(content, source, ktype)
                return f"Added knowledge from {source}"
            return "Please fill all fields"
        
        def search_kb(query):
            if query:
                results = semantic_search_with_filters(query, n_results=3)
                if results and results['documents']:
                    formatted = ""
                    for i, doc in enumerate(results['documents'][0]):
                        formatted += f"Result {i+1}:\n{doc[:200]}...\n\n"
                    return formatted
                return "No results found"
            return "Please enter a search query"
        
        msg.submit(respond, [msg, chatbot], [chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
        stats_btn.click(show_stats, outputs=stats_output)
        add_knowledge_btn.click(add_knowledge, [custom_content, custom_source, custom_type], outputs=stats_output)
        search_btn.click(search_kb, search_query, search_results)
    
    return demo

# Create and launch the enhanced interface (optional)
# enhanced_demo = create_enhanced_interface()
# enhanced_demo.launch()
```

## Summary of RAG Implementation

This RAG system provides:

1. **Document Processing**: Automatic chunking of LinkedIn profile and summary
2. **Vector Storage**: ChromaDB with persistent storage and metadata
3. **Semantic Search**: Context-aware retrieval based on user queries
4. **Enhanced Chat**: Integration with existing chatbot and tool system
5. **Custom Knowledge**: Ability to add new information dynamically
6. **Analytics**: Monitoring and statistics for the knowledge base
7. **Advanced Interface**: Optional enhanced Gradio interface with RAG capabilities

### Key Benefits:
- **Better Context**: Retrieves relevant information before generating responses
- **Scalable**: Can handle large amounts of career-related information
- **Dynamic**: Allows adding new knowledge on the fly
- **Searchable**: Users can search the knowledge base directly
- **Analytics**: Monitor the system's performance and content
- **Integration**: Works seamlessly with existing tools and functions

### How to Implement:
1. **Copy sections 1-4** to set up the RAG system
2. **Replace your existing `chat` function** with the enhanced version from section 5
3. **Keep using the same interface** - no changes to your Gradio setup needed
4. **Optional**: Add sections 6-8 for advanced features and enhanced interface

### Next Steps:
1. Test the RAG system with various career-related queries
2. Add more custom knowledge about your specific skills and experiences
3. Fine-tune chunk sizes and overlap for optimal retrieval
4. Consider adding more document sources (resume, portfolio, etc.)
5. Implement feedback mechanisms to improve retrieval quality
